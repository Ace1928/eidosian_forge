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
def hdi(ary, hdi_prob=None, circular=False, multimodal=False, skipna=False, group='posterior', var_names=None, filter_vars=None, coords=None, max_modes=10, dask_kwargs=None, **kwargs):
    """
    Calculate highest density interval (HDI) of array for given probability.

    The HDI is the minimum width Bayesian credible interval (BCI).

    Parameters
    ----------
    ary: obj
        object containing posterior samples.
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
    hdi_prob: float, optional
        Prob for which the highest density interval will be computed. Defaults to
        ``stats.hdi_prob`` rcParam.
    circular: bool, optional
        Whether to compute the hdi taking into account `x` is a circular variable
        (in the range [-np.pi, np.pi]) or not. Defaults to False (i.e non-circular variables).
        Only works if multimodal is False.
    multimodal: bool, optional
        If true it may compute more than one hdi if the distribution is multimodal and the
        modes are well separated.
    skipna: bool, optional
        If true ignores nan values when computing the hdi. Defaults to false.
    group: str, optional
        Specifies which InferenceData group should be used to calculate hdi.
        Defaults to 'posterior'
    var_names: list, optional
        Names of variables to include in the hdi report. Prefix the variables by ``~``
        when you want to exclude them from the report: `["~beta"]` instead of `["beta"]`
        (see :func:`arviz.summary` for more details).
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        ``pandas.filter``.
    coords: mapping, optional
        Specifies the subset over to calculate hdi.
    max_modes: int, optional
        Specifies the maximum number of modes for multimodal case.
    dask_kwargs : dict, optional
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.
    kwargs: dict, optional
        Additional keywords passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    np.ndarray or xarray.Dataset, depending upon input
        lower(s) and upper(s) values of the interval(s).

    See Also
    --------
    plot_hdi : Plot highest density intervals for regression data.
    xarray.Dataset.quantile : Calculate quantiles of array for given probabilities.

    Examples
    --------
    Calculate the HDI of a Normal random variable:

    .. ipython::

        In [1]: import arviz as az
           ...: import numpy as np
           ...: data = np.random.normal(size=2000)
           ...: az.hdi(data, hdi_prob=.68)

    Calculate the HDI of a dataset:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data('centered_eight')
           ...: az.hdi(data)

    We can also calculate the HDI of some of the variables of dataset:

    .. ipython::

        In [1]: az.hdi(data, var_names=["mu", "theta"])

    By default, ``hdi`` is calculated over the ``chain`` and ``draw`` dimensions. We can use the
    ``input_core_dims`` argument of :func:`~arviz.wrap_xarray_ufunc` to change this. In this example
    we calculate the HDI also over the ``school`` dimension:

    .. ipython::

        In [1]: az.hdi(data, var_names="theta", input_core_dims = [["chain","draw", "school"]])

    We can also calculate the hdi over a particular selection:

    .. ipython::

        In [1]: az.hdi(data, coords={"chain":[0, 1, 3]}, input_core_dims = [["draw"]])

    """
    if hdi_prob is None:
        hdi_prob = rcParams['stats.hdi_prob']
    elif not 1 >= hdi_prob > 0:
        raise ValueError('The value of hdi_prob should be in the interval (0, 1]')
    func_kwargs = {'hdi_prob': hdi_prob, 'skipna': skipna, 'out_shape': (max_modes, 2) if multimodal else (2,)}
    kwargs.setdefault('output_core_dims', [['mode', 'hdi'] if multimodal else ['hdi']])
    if not multimodal:
        func_kwargs['circular'] = circular
    else:
        func_kwargs['max_modes'] = max_modes
    func = _hdi_multimodal if multimodal else _hdi
    isarray = isinstance(ary, np.ndarray)
    if isarray and ary.ndim <= 1:
        func_kwargs.pop('out_shape')
        hdi_data = func(ary, **func_kwargs)
        return hdi_data[~np.isnan(hdi_data).all(axis=1), :] if multimodal else hdi_data
    if isarray and ary.ndim == 2:
        warnings.warn('hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions', FutureWarning, stacklevel=2)
        ary = np.expand_dims(ary, 0)
    ary = convert_to_dataset(ary, group=group)
    if coords is not None:
        ary = get_coords(ary, coords)
    var_names = _var_names(var_names, ary, filter_vars)
    ary = ary[var_names] if var_names else ary
    hdi_coord = xr.DataArray(['lower', 'higher'], dims=['hdi'], attrs=dict(hdi_prob=hdi_prob))
    hdi_data = _wrap_xarray_ufunc(func, ary, func_kwargs=func_kwargs, dask_kwargs=dask_kwargs, **kwargs).assign_coords({'hdi': hdi_coord})
    hdi_data = hdi_data.dropna('mode', how='all') if multimodal else hdi_data
    return hdi_data.x.values if isarray else hdi_data