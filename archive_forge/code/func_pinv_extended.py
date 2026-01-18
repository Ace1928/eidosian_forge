import numpy as np
import pandas as pd
import scipy.linalg
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import array_like
def pinv_extended(x, rcond=1e-15):
    """
    Return the pinv of an array X as well as the singular values
    used in computation.

    Code adapted from numpy.
    """
    x = np.asarray(x)
    x = x.conjugate()
    u, s, vt = np.linalg.svd(x, False)
    s_orig = np.copy(s)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond * np.maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1.0 / s[i]
        else:
            s[i] = 0.0
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis], np.transpose(u)))
    return (res, s_orig)