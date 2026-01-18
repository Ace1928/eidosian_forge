from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from scipy import sparse
from statsmodels.tools.grouputils import (dummy_sparse, Grouping, Group,
from statsmodels.datasets import grunfeld, anes96
@pytest.mark.smoke
def test_group_sums():
    g = np.array([0, 0, 1, 2, 1, 1, 2, 0])
    group_sums(np.arange(len(g) * 3 * 2).reshape(len(g), 3, 2), g, use_bincount=False).T
    group_sums(np.arange(len(g) * 3 * 2).reshape(len(g), 3, 2)[:, :, 0], g)
    group_sums(np.arange(len(g) * 3 * 2).reshape(len(g), 3, 2)[:, :, 1], g)