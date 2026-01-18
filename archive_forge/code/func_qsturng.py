from statsmodels.compat.python import lrange
import math
import scipy.stats
import numpy as np
from scipy.optimize import fminbound
def qsturng(p, r, v):
    """Approximates the quantile p for a studentized range
       distribution having v degrees of freedom and r samples
       for probability p.

    Parameters
    ----------
    p : (scalar, array_like)
        The cumulative probability value
        p >= .1 and p <=.999
        (values under .5 are not recommended)
    r : (scalar, array_like)
        The number of samples
        r >= 2 and r <= 200
        (values over 200 are permitted but not recommended)
    v : (scalar, array_like)
        The sample degrees of freedom
        if p >= .9:
            v >=1 and v >= inf
        else:
            v >=2 and v >= inf

    Returns
    -------
    q : (scalar, array_like)
        approximation of the Studentized Range
    """
    if all(map(_isfloat, [p, r, v])):
        return _qsturng(p, r, v)
    return _vqsturng(p, r, v)