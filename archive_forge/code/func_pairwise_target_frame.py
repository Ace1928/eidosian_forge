import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.fixture
def pairwise_target_frame():
    """Pairwise target frame for test_pairwise"""
    return DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[0, 1])