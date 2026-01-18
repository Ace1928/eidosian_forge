from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from scipy import sparse
from statsmodels.tools.grouputils import (dummy_sparse, Grouping, Group,
from statsmodels.datasets import grunfeld, anes96
@pytest.mark.smoke
def test_group_class():
    g = np.array([0, 0, 1, 2, 1, 1, 2, 0])
    x = np.arange(len(g) * 3).reshape(len(g), 3, order='F')
    mygroup = Group(g)
    mygroup.group_int
    mygroup.group_sums(x)
    mygroup.labels()