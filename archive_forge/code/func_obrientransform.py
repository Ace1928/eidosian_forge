import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def obrientransform(*args):
    """
    Computes a transform on input data (any number of columns).  Used to
    test for homogeneity of variance prior to running one-way stats.  Each
    array in ``*args`` is one level of a factor.  If an `f_oneway()` run on
    the transformed data and found significant, variances are unequal.   From
    Maxwell and Delaney, p.112.

    Returns: transformed data for use in an ANOVA
    """
    data = argstoarray(*args).T
    v = data.var(axis=0, ddof=1)
    m = data.mean(0)
    n = data.count(0).astype(float)
    data -= m
    data **= 2
    data *= (n - 1.5) * n
    data -= 0.5 * v * (n - 1)
    data /= (n - 1.0) * (n - 2.0)
    if not ma.allclose(v, data.mean(0)):
        raise ValueError('Lack of convergence in obrientransform.')
    return data