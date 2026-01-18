from itertools import product
import numpy as np
import pytest
from pandas._libs import (
from pandas import (
import pandas._testing as tm
@pytest.fixture
def idx_dup():
    major_axis = Index(['foo', 'bar', 'baz', 'qux'])
    minor_axis = Index(['one', 'two'])
    major_codes = np.array([0, 0, 1, 0, 1, 1])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index_names = ['first', 'second']
    mi = MultiIndex(levels=[major_axis, minor_axis], codes=[major_codes, minor_codes], names=index_names, verify_integrity=False)
    return mi