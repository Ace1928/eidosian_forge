import itertools
import warnings
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Mapping, cast, Callable
import numpy as np
import pandas as pd
import scipy.stats as st
from xarray_einstats import stats
import xarray as xr
from scipy.optimize import minimize
from typing_extensions import Literal
from .. import _log
from ..data import InferenceData, convert_to_dataset, convert_to_inference_data, extract
from ..rcparams import rcParams, ScaleKeyword, ICKeyword
from ..utils import Numba, _numba_var, _var_names, get_coords
from .density_utils import get_bins as _get_bins
from .density_utils import histogram as _histogram
from .density_utils import kde as _kde
from .diagnostics import _mc_error, _multichain_statistics, ess
from .stats_utils import ELPDData, _circular_standard_deviation, smooth_data
from .stats_utils import get_log_likelihood as _get_log_likelihood
from .stats_utils import get_log_prior as _get_log_prior
from .stats_utils import logsumexp as _logsumexp
from .stats_utils import make_ufunc as _make_ufunc
from .stats_utils import stats_variance_2d as svar
from .stats_utils import wrap_xarray_ufunc as _wrap_xarray_ufunc
from ..sel_utils import xarray_var_iter
from ..labels import BaseLabeller
def psislw(log_weights, reff=1.0):
    """
    Pareto smoothed importance sampling (PSIS).

    Notes
    -----
    If the ``log_weights`` input is an :class:`~xarray.DataArray` with a dimension
    named ``__sample__`` (recommended) ``psislw`` will interpret this dimension as samples,
    and all other dimensions as dimensions of the observed data, looping over them to
    calculate the psislw of each observation. If no ``__sample__`` dimension is present or
    the input is a numpy array, the last dimension will be interpreted as ``__sample__``.

    Parameters
    ----------
    log_weights : DataArray or (..., N) array-like
        Array of size (n_observations, n_samples)
    reff : float, default 1
        relative MCMC efficiency, ``ess / n``

    Returns
    -------
    lw_out : DataArray or (..., N) ndarray
        Smoothed, truncated and normalized log weights.
    kss : DataArray or (...) ndarray
        Estimates of the shape parameter *k* of the generalized Pareto
        distribution.

    References
    ----------
    * Vehtari et al. (2015) see https://arxiv.org/abs/1507.02646

    See Also
    --------
    loo : Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).

    Examples
    --------
    Get Pareto smoothed importance sampling (PSIS) log weights:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("non_centered_eight")
           ...: log_likelihood = data.log_likelihood["obs"].stack(
           ...:     __sample__=["chain", "draw"]
           ...: )
           ...: az.psislw(-log_likelihood, reff=0.8)

    """
    if hasattr(log_weights, '__sample__'):
        n_samples = len(log_weights.__sample__)
        shape = [size for size, dim in zip(log_weights.shape, log_weights.dims) if dim != '__sample__']
    else:
        n_samples = log_weights.shape[-1]
        shape = log_weights.shape[:-1]
    cutoff_ind = -int(np.ceil(min(n_samples / 5.0, 3 * (n_samples / reff) ** 0.5))) - 1
    cutoffmin = np.log(np.finfo(float).tiny)
    out = (np.empty_like(log_weights), np.empty(shape))
    func_kwargs = {'cutoff_ind': cutoff_ind, 'cutoffmin': cutoffmin, 'out': out}
    ufunc_kwargs = {'n_dims': 1, 'n_output': 2, 'ravel': False, 'check_shape': False}
    kwargs = {'input_core_dims': [['__sample__']], 'output_core_dims': [['__sample__'], []]}
    log_weights, pareto_shape = _wrap_xarray_ufunc(_psislw, log_weights, ufunc_kwargs=ufunc_kwargs, func_kwargs=func_kwargs, **kwargs)
    if isinstance(log_weights, xr.DataArray):
        log_weights = log_weights.rename('log_weights')
    if isinstance(pareto_shape, xr.DataArray):
        pareto_shape = pareto_shape.rename('pareto_shape')
    return (log_weights, pareto_shape)