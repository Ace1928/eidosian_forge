import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def rank_omit(a, method, axis):
    return np.apply_along_axis(lambda a: rank_1d_omit(a, method), axis, a)