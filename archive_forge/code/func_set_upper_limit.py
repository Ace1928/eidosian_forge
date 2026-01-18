import warnings
import numpy as np
import pandas as pd
from scipy import stats
def set_upper_limit(cohn):
    """ Sets the upper_dl DL for each row of the Cohn dataframe. """
    if cohn.shape[0] > 1:
        return cohn['lower_dl'].shift(-1).fillna(value=np.inf)
    else:
        return [np.inf]