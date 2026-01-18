from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp
def logbasechange(a, b):
    """
    There is a one-to-one transformation of the entropy value from
    a log base b to a log base a :

    H_{b}(X)=log_{b}(a)[H_{a}(X)]

    Returns
    -------
    log_{b}(a)
    """
    return np.log(b) / np.log(a)