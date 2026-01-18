import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs, uniquecols
from ..doctools import document
from ..exceptions import PlotnineError
from .stat import stat
def median_hilow(series, confidence_interval=0.95):
    """
    Median and a selected pair of outer quantiles having equal tail areas

    Parameters
    ----------
    series : pandas.Series
        Values
    confidence_interval : float
        Confidence interval in the range (0, 1).
    """
    tail = (1 - confidence_interval) / 2
    return pd.DataFrame({'y': [np.median(series)], 'ymin': np.percentile(series, 100 * tail), 'ymax': np.percentile(series, 100 * (1 - tail))})