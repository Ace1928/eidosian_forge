import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs, uniquecols
from ..doctools import document
from ..exceptions import PlotnineError
from .stat import stat
def mean_sdl(series, mult=2):
    """
    Mean +/- a constant times the standard deviation

    Parameters
    ----------
    series : pandas.Series
        Values
    mult : float
        Multiplication factor.
    """
    m = series.mean()
    s = series.std()
    return pd.DataFrame({'y': [m], 'ymin': m - mult * s, 'ymax': m + mult * s})