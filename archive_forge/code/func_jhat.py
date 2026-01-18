import contextlib
import functools
import operator
import warnings
import numpy as np
from numpy.core import overrides
def jhat(nbins):
    hh = ptp_x / nbins
    p_k = np.histogram(x, bins=nbins, range=range)[0] / n
    return (2 - (n + 1) * p_k.dot(p_k)) / hh