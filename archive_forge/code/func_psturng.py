from statsmodels.compat.python import lrange
import math
import scipy.stats
import numpy as np
from scipy.optimize import fminbound
def psturng(q, r, v):
    """Evaluates the probability from 0 to q for a studentized
       range having v degrees of freedom and r samples.

    Parameters
    ----------
    q : (scalar, array_like)
        quantile value of Studentized Range
        q >= 0.
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
    p : (scalar, array_like)
        1. - area from zero to q under the Studentized Range
        distribution. When v == 1, p is bound between .001
        and .1, when v > 1, p is bound between .001 and .9.
        Values between .5 and .9 are 1st order appoximations.
    """
    if all(map(_isfloat, [q, r, v])):
        return _psturng(q, r, v)
    return _vpsturng(q, r, v)