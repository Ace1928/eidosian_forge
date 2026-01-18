from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.evaluation import after_stat
from .stat import stat
def compute_density(x, weight, range, **params):
    """
    Compute density
    """
    import statsmodels.api as sm
    x = np.asarray(x, dtype=float)
    not_nan = ~np.isnan(x)
    x = x[not_nan]
    bw = params['bw']
    kernel = params['kernel']
    n = len(x)
    assert isinstance(bw, (str, float))
    if n == 0 or (n == 1 and isinstance(bw, str)):
        if n == 1:
            warn('To compute the density of a group with only one value set the bandwidth manually. e.g `bw=0.1`', PlotnineWarning)
        warn('Groups with fewer than 2 data points have been removed.', PlotnineWarning)
        return pd.DataFrame()
    if weight is None:
        if kernel != 'gau':
            weight = np.ones(n) / n
    else:
        weight = np.asarray(weight, dtype=float)
    fft = kernel == 'gau' and weight is None
    if bw == 'nrd0':
        bw = nrd0(x)
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit(kernel=kernel, bw=bw, fft=fft, weights=weight, adjust=params['adjust'], cut=params['cut'], gridsize=params['gridsize'], clip=params['clip'])
    x2 = np.linspace(range[0], range[1], params['n'])
    try:
        y = kde.evaluate(x2)
        if np.isscalar(y) and np.isnan(y):
            raise ValueError('kde.evaluate returned nan')
    except ValueError:
        y = []
        for _x in x2:
            result = kde.evaluate(_x)
            if isinstance(result, (float, int)):
                y.append(result)
            else:
                y.append(result[0])
    y = np.asarray(y)
    not_nan = ~np.isnan(y)
    x2 = x2[not_nan]
    y = y[not_nan]
    return pd.DataFrame({'x': x2, 'density': y, 'scaled': y / np.max(y) if len(y) else [], 'count': y * n, 'n': n})