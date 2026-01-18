from __future__ import annotations
import typing
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import groupby_apply
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.evaluation import after_stat
from .binning import (
from .stat import stat
def densitybin(x, weight: FloatArrayLike | None, binwidth: float | None, bins: int=30, rangee: Optional[tuple[float, float]]=None):
    """
    Do density binning

    It does not collapse each bin with a count.

    Parameters
    ----------
    x : array_like
        Numbers to bin
    weight : array_like
        Weights
    binwidth : numeric
        Size of the bins
    bins : int
        Number of bins
    rangee : tuple
        Range of x

    Returns
    -------
    data : DataFrame
    """
    if all(pd.isna(x)):
        return pd.DataFrame()
    if weight is None:
        weight = np.ones(len(x))
    weight = np.asarray(weight)
    weight[np.isnan(weight)] = 0
    if rangee is None:
        rangee = (np.min(x), np.max(x))
    if bins is None:
        bins = 30
    if binwidth is None:
        binwidth = np.ptp(rangee) / bins
    order = np.argsort(x)
    weight = weight[order]
    x = x[order]
    cbin = 0
    bin_ids = []
    binend = -np.inf
    for value in x:
        if value >= binend:
            binend = value + binwidth
            cbin = cbin + 1
        bin_ids.append(cbin)

    def func(series):
        return (series.min() + series.max()) / 2
    results = pd.DataFrame({'x': x, 'bin': bin_ids, 'binwidth': binwidth, 'weight': weight})
    results['bincenter'] = results.groupby('bin')['x'].transform(func)
    return results