import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from scipy.stats import CensoredData
def test_count_censored(self):
    x = [1, 2, 3]
    data1 = CensoredData(x)
    assert data1.num_censored() == 0
    data2 = CensoredData(uncensored=[2.5], left=[10], interval=[[0, 1]])
    assert data2.num_censored() == 2