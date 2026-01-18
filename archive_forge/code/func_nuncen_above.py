import warnings
import numpy as np
import pandas as pd
from scipy import stats
def nuncen_above(row):
    """ A, the number of uncensored obs above the given threshold.
        """
    above = df[observations] >= row['lower_dl']
    below = df[observations] < row['upper_dl']
    detect = ~df[censorship]
    return df[above & below & detect].shape[0]