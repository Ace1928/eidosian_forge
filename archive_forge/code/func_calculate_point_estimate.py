import importlib
import warnings
from typing import Any, Dict
import matplotlib as mpl
import numpy as np
import packaging
from matplotlib.colors import to_hex
from scipy.stats import mode, rankdata
from scipy.interpolate import CubicSpline
from ..rcparams import rcParams
from ..stats.density_utils import kde
from ..stats import hdi
def calculate_point_estimate(point_estimate, values, bw='default', circular=False, skipna=False):
    """Validate and calculate the point estimate.

    Parameters
    ----------
    point_estimate : Optional[str]
        Plot point estimate per variable. Values should be 'mean', 'median', 'mode' or None.
        Defaults to 'auto' i.e. it falls back to default set in rcParams.
    values : 1-d array
    bw: Optional[float or str]
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "isj" or "experimental" when `circular` is False
        and "taylor" (for now) when `circular` is True.
        Defaults to "default" which means "experimental" when variable is not circular
        and "taylor" when it is.
    circular: Optional[bool]
        If True, it interprets the values passed are from a circular variable measured in radians
        and a circular KDE is used. Only valid for 1D KDE. Defaults to False.
    skipna=True,
        If true ignores nan values when computing the hdi. Defaults to false.

    Returns
    -------
    point_value : float
        best estimate of data distribution
    """
    point_value = None
    if point_estimate == 'auto':
        point_estimate = rcParams['plot.point_estimate']
    elif point_estimate not in ('mean', 'median', 'mode', None):
        raise ValueError(f"Point estimate should be 'mean', 'median', 'mode' or None, not {point_estimate}")
    if point_estimate == 'mean':
        point_value = np.nanmean(values) if skipna else np.mean(values)
    elif point_estimate == 'mode':
        if values.dtype.kind == 'f':
            if bw == 'default':
                bw = 'taylor' if circular else 'experimental'
            x, density = kde(values, circular=circular, bw=bw)
            point_value = x[np.argmax(density)]
        else:
            point_value = int(mode(values).mode)
    elif point_estimate == 'median':
        point_value = np.nanmedian(values) if skipna else np.median(values)
    return point_value