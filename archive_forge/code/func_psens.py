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
def psens(data, *, component='prior', component_var_names=None, component_coords=None, var_names=None, coords=None, filter_vars=None, delta=0.01, dask_kwargs=None):
    """Compute power-scaling sensitivity diagnostic.

    Power-scales the prior or likelihood and calculates how much the posterior is affected.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
        For ndarray: shape = (chain, draw).
        For n-dimensional ndarray transform first to dataset with ``az.convert_to_dataset``.
    component : {"prior", "likelihood"}, default "prior"
        When `component` is "likelihood", the log likelihood values are retrieved
        from the ``log_likelihood`` group as pointwise log likelihood and added
        together. With "prior", the log prior values are retrieved from the
        ``log_prior`` group.
    component_var_names : str, optional
        Name of the prior or log likelihood variables to use
    component_coords : dict, optional
        Coordinates defining a subset over the component element for which to
        compute the prior sensitivity diagnostic.
    var_names : list of str, optional
        Names of posterior variables to include in the power scaling sensitivity diagnostic
    coords : dict, optional
        Coordinates defining a subset over the posterior. Only these variables will
        be used when computing the prior sensitivity.
    filter_vars: {None, "like", "regex"}, default None
        If ``None`` (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    delta : float
        Value for finite difference derivative calculation.
    dask_kwargs : dict, optional
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    xarray.Dataset
        Returns dataset of power-scaling sensitivity diagnostic values.
        Higher sensitivity values indicate greater sensitivity.
        Prior sensitivity above 0.05 indicates informative prior.
        Likelihood sensitivity below 0.05 indicates weak or nonin-formative likelihood.

    Examples
    --------
    Compute the likelihood sensitivity for the non centered eight model:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("non_centered_eight")
           ...: az.psens(data, component="likelihood")

    To compute the prior sensitivity, we need to first compute the log prior
    at each posterior sample. In our case, we know mu has a normal prior :math:`N(0, 5)`,
    tau is a half cauchy prior with scale/beta parameter 5,
    and theta has a standard normal as prior.
    We add this information to the ``log_prior`` group before computing powerscaling
    check with ``psens``

    .. ipython::

        In [1]: from xarray_einstats.stats import XrContinuousRV
           ...: from scipy.stats import norm, halfcauchy
           ...: post = data.posterior
           ...: log_prior = {
           ...:     "mu": XrContinuousRV(norm, 0, 5).logpdf(post["mu"]),
           ...:     "tau": XrContinuousRV(halfcauchy, scale=5).logpdf(post["tau"]),
           ...:     "theta_t": XrContinuousRV(norm, 0, 1).logpdf(post["theta_t"]),
           ...: }
           ...: data.add_groups({"log_prior": log_prior})
           ...: az.psens(data, component="prior")

    Notes
    -----
    The diagnostic is computed by power-scaling the specified component (prior or likelihood)
    and determining the degree to which the posterior changes as described in [1]_.
    It uses Pareto-smoothed importance sampling to avoid refitting the model.

    References
    ----------
    .. [1] Kallioinen et al, *Detecting and diagnosing prior and likelihood sensitivity with
       power-scaling*, 2022, https://arxiv.org/abs/2107.14054

    """
    dataset = extract(data, var_names=var_names, filter_vars=filter_vars, group='posterior')
    if coords is None:
        dataset = dataset.sel(coords)
    if component == 'likelihood':
        component_draws = _get_log_likelihood(data, var_name=component_var_names, single_var=False)
    elif component == 'prior':
        component_draws = _get_log_prior(data, var_names=component_var_names)
    else:
        raise ValueError('Value for `component` argument not recognized')
    component_draws = component_draws.stack(__sample__=('chain', 'draw'))
    if component_coords is None:
        component_draws = component_draws.sel(component_coords)
    if isinstance(component_draws, xr.DataArray):
        component_draws = component_draws.to_dataset()
    if len(component_draws.dims):
        component_draws = component_draws.to_stacked_array('latent-obs_var', sample_dims=('__sample__',)).sum('latent-obs_var')
    lower_alpha = 1 / (1 + delta)
    upper_alpha = 1 + delta
    lower_w = np.exp(_powerscale_lw(component_draws=component_draws, alpha=lower_alpha))
    lower_w = lower_w / np.sum(lower_w)
    upper_w = np.exp(_powerscale_lw(component_draws=component_draws, alpha=upper_alpha))
    upper_w = upper_w / np.sum(upper_w)
    ufunc_kwargs = {'n_dims': 1, 'ravel': False}
    func_kwargs = {'lower_weights': lower_w.values, 'upper_weights': upper_w.values, 'delta': delta}
    return _wrap_xarray_ufunc(_powerscale_sens, dataset, ufunc_kwargs=ufunc_kwargs, func_kwargs=func_kwargs, dask_kwargs=dask_kwargs, input_core_dims=[['sample']])