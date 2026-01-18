from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from ..exceptions import PlotnineError
from ..scales.scale_discrete import scale_discrete
def freedman_diaconis_bins(a):
    """
    Calculate number of hist bins using Freedman-Diaconis rule.
    """
    from scipy.stats import iqr
    a = np.asarray(a)
    h = 2 * iqr(a, nan_policy='omit') / len(a) ** (1 / 3)
    if h == 0:
        bins = np.ceil(np.sqrt(a.size))
    else:
        bins = np.ceil((np.nanmax(a) - np.nanmin(a)) / h)
    return int(bins)