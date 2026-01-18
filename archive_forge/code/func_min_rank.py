import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def min_rank(a):
    return [1 + sum((i < j for i in a)) for j in a]