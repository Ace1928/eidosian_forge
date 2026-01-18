import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs, uniquecols
from ..doctools import document
from ..exceptions import PlotnineError
from .stat import stat
def mean_cl_boot(series, n_samples=1000, confidence_interval=0.95, random_state=None):
    """
    Bootstrapped mean with confidence interval

    Parameters
    ----------
    series : pandas.Series
        Values
    n_samples : int, default=1000
        Number of sample to draw.
    confidence_interval : float
        Confidence interval in the range (0, 1).
    random_state : int | ~numpy.random.RandomState, default=None
        Seed or Random number generator to use. If `None`, then
        numpy global generator [](`numpy.random`) is used.
    """
    return bootstrap_statistics(series, np.mean, n_samples=n_samples, confidence_interval=confidence_interval, random_state=random_state)