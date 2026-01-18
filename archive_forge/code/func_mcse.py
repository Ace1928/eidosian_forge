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
def mcse(data, *, var_names=None, method='mean', prob=None, dask_kwargs=None):
    """Calculate Markov Chain Standard Error statistic.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an :class:`arviz.InferenceData` object
        Refer to documentation of :func:`arviz.convert_to_dataset` for details
        For ndarray: shape = (chain, draw).
        For n-dimensional ndarray transform first to dataset with ``az.convert_to_dataset``.
    var_names : list
        Names of variables to include in the rhat report
    method : str
        Select mcse method. Valid methods are:
        - "mean"
        - "sd"
        - "median"
        - "quantile"

    prob : float
        Quantile information.
    dask_kwargs : dict, optional
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    xarray.Dataset
        Return the msce dataset

    See Also
    --------
    ess : Compute autocovariance estimates for every lag for the input array.
    summary : Create a data frame with summary statistics.
    plot_mcse : Plot quantile or local Monte Carlo Standard Error.

    Examples
    --------
    Calculate the Markov Chain Standard Error using the default arguments:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("non_centered_eight")
           ...: az.mcse(data)

    Calculate the Markov Chain Standard Error using the quantile method:

    .. ipython::

        In [1]: az.mcse(data, method="quantile", prob=0.7)

    """
    methods = {'mean': _mcse_mean, 'sd': _mcse_sd, 'median': _mcse_median, 'quantile': _mcse_quantile}
    if method not in methods:
        raise TypeError('mcse method {} not found. Valid methods are:\n{}'.format(method, '\n    '.join(methods)))
    mcse_func = methods[method]
    if method == 'quantile' and prob is None:
        raise TypeError('Quantile (prob) information needs to be defined.')
    if isinstance(data, np.ndarray):
        data = np.atleast_2d(data)
        if len(data.shape) < 3:
            if prob is not None:
                return mcse_func(data, prob=prob)
            return mcse_func(data)
        msg = 'Only uni-dimensional ndarray variables are supported. Please transform first to dataset with `az.convert_to_dataset`.'
        raise TypeError(msg)
    dataset = convert_to_dataset(data, group='posterior')
    var_names = _var_names(var_names, dataset)
    dataset = dataset if var_names is None else dataset[var_names]
    ufunc_kwargs = {'ravel': False}
    func_kwargs = {} if prob is None else {'prob': prob}
    return _wrap_xarray_ufunc(mcse_func, dataset, ufunc_kwargs=ufunc_kwargs, func_kwargs=func_kwargs, dask_kwargs=dask_kwargs)