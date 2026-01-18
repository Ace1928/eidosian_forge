from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from ..exceptions import PlotnineError
from ..scales.scale_discrete import scale_discrete
def result_dataframe(count, x, width, xmin=None, xmax=None):
    """
    Create a dataframe to hold bin information
    """
    if xmin is None:
        xmin = x - width / 2
    if xmax is None:
        xmax = x + width / 2
    xmin[1:] = xmax[:-1]
    density = count / width / np.sum(np.abs(count))
    out = pd.DataFrame({'count': count, 'x': x, 'xmin': xmin, 'xmax': xmax, 'width': width, 'density': density, 'ncount': count / np.max(np.abs(count)), 'ndensity': density / np.max(np.abs(density)), 'ngroup:': np.sum(np.abs(count))})
    return out