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
def scoreatpercentile(data, per, limit=(), alphap=0.4, betap=0.4):
    """Calculate the score at the given 'per' percentile of the
    sequence a.  For example, the score at per=50 is the median.

    This function is a shortcut to mquantile

    """
    if per < 0 or per > 100.0:
        raise ValueError('The percentile should be between 0. and 100. ! (got %s)' % per)
    return mquantiles(data, prob=[per / 100.0], alphap=alphap, betap=betap, limit=limit, axis=0).squeeze()