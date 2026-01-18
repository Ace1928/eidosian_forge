from itertools import permutations
import numpy as np
import pytest
from pandas._libs.interval import IntervalTree
from pandas.compat import IS64
import pandas._testing as tm
def skipif_32bit(param):
    """
    Skip parameters in a parametrize on 32bit systems. Specifically used
    here to skip leaf_size parameters related to GH 23440.
    """
    marks = pytest.mark.skipif(not IS64, reason='GH 23440: int type mismatch on 32bit')
    return pytest.param(param, marks=marks)