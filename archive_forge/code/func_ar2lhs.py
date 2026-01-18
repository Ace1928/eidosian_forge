import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def ar2lhs(ar):
    """convert full (rhs) lagpolynomial into a reduced, left side lagpoly array

    this is mainly a reminder about the definition
    """
    return -ar[1:]