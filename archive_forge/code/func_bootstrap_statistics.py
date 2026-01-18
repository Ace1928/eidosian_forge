import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs, uniquecols
from ..doctools import document
from ..exceptions import PlotnineError
from .stat import stat
def bootstrap_statistics(series, statistic, n_samples=1000, confidence_interval=0.95, random_state=None):
    """
    Default parameters taken from
    R's Hmisc smean.cl.boot
    """
    if random_state is None:
        random_state = np.random
    alpha = 1 - confidence_interval
    size = (n_samples, len(series))
    inds = random_state.randint(0, len(series), size=size)
    samples = series.to_numpy()[inds]
    means = np.sort(statistic(samples, axis=1))
    return pd.DataFrame({'ymin': means[int(alpha / 2 * n_samples)], 'ymax': means[int((1 - alpha / 2) * n_samples)], 'y': [statistic(series)]})