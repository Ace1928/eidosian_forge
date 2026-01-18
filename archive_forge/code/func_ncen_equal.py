import warnings
import numpy as np
import pandas as pd
from scipy import stats
def ncen_equal(row):
    """ C, the number of censored observations at the given
        threshold.
        """
    censored_index = df[censorship]
    censored_data = df[observations][censored_index]
    censored_below = censored_data == row['lower_dl']
    return censored_below.sum()