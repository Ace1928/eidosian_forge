import warnings
from collections.abc import Sequence
import numpy as np
import packaging
import pandas as pd
import scipy
from scipy import stats
from ..data import convert_to_dataset
from ..utils import Numba, _numba_var, _stack, _var_names
from .density_utils import histogram as _histogram
from .stats_utils import _circular_standard_deviation, _sqrt
from .stats_utils import autocov as _autocov
from .stats_utils import not_valid as _not_valid
from .stats_utils import quantile as _quantile
from .stats_utils import stats_variance_2d as svar
from .stats_utils import wrap_xarray_ufunc as _wrap_xarray_ufunc
def rhat(data, *, var_names=None, method='rank', dask_kwargs=None):
    """Compute estimate of rank normalized splitR-hat for a set of traces.

    The rank normalized R-hat diagnostic tests for lack of convergence by comparing the variance
    between multiple chains to the variance within each chain. If convergence has been achieved,
    the between-chain and within-chain variances should be identical. To be most effective in
    detecting evidence for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
        For ndarray: shape = (chain, draw).
        For n-dimensional ndarray transform first to dataset with ``az.convert_to_dataset``.
    var_names : list
        Names of variables to include in the rhat report
    method : str
        Select R-hat method. Valid methods are:
        - "rank"        # recommended by Vehtari et al. (2019)
        - "split"
        - "folded"
        - "z_scale"
        - "identity"
    dask_kwargs : dict, optional
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    xarray.Dataset
      Returns dataset of the potential scale reduction factors, :math:`\\hat{R}`

    See Also
    --------
    ess : Calculate estimate of the effective sample size (ess).
    mcse : Calculate Markov Chain Standard Error statistic.
    plot_forest : Forest plot to compare HDI intervals from a number of distributions.

    Notes
    -----
    The diagnostic is computed by:

      .. math:: \\hat{R} = \\frac{\\hat{V}}{W}

    where :math:`W` is the within-chain variance and :math:`\\hat{V}` is the posterior variance
    estimate for the pooled rank-traces. This is the potential scale reduction factor, which
    converges to unity when each of the traces is a sample from the target posterior. Values
    greater than one indicate that one or more chains have not yet converged.

    Rank values are calculated over all the chains with ``scipy.stats.rankdata``.
    Each chain is split in two and normalized with the z-transform following Vehtari et al. (2019).

    References
    ----------
    * Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008
    * Gelman et al. BDA (2014)
    * Brooks and Gelman (1998)
    * Gelman and Rubin (1992)

    Examples
    --------
    Calculate the R-hat using the default arguments:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("non_centered_eight")
           ...: az.rhat(data)

    Calculate the R-hat of some variables using the folded method:

    .. ipython::

        In [1]: az.rhat(data, var_names=["mu", "theta_t"], method="folded")

    """
    methods = {'rank': _rhat_rank, 'split': _rhat_split, 'folded': _rhat_folded, 'z_scale': _rhat_z_scale, 'identity': _rhat_identity}
    if method not in methods:
        raise TypeError(f'R-hat method {method} not found. Valid methods are:\n{', '.join(methods)}')
    rhat_func = methods[method]
    if isinstance(data, np.ndarray):
        data = np.atleast_2d(data)
        if len(data.shape) < 3:
            return rhat_func(data)
        msg = 'Only uni-dimensional ndarray variables are supported. Please transform first to dataset with `az.convert_to_dataset`.'
        raise TypeError(msg)
    dataset = convert_to_dataset(data, group='posterior')
    var_names = _var_names(var_names, dataset)
    dataset = dataset if var_names is None else dataset[var_names]
    ufunc_kwargs = {'ravel': False}
    func_kwargs = {}
    return _wrap_xarray_ufunc(rhat_func, dataset, ufunc_kwargs=ufunc_kwargs, func_kwargs=func_kwargs, dask_kwargs=dask_kwargs)