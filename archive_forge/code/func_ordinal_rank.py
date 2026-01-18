import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def ordinal_rank(a):
    return min_rank([(x, i) for i, x in enumerate(a)])