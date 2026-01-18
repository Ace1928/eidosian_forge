from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from scipy import sparse
from statsmodels.tools.grouputils import (dummy_sparse, Grouping, Group,
from statsmodels.datasets import grunfeld, anes96
def test_dummy_sparse(self):
    data = self.data
    self.grouping.dummy_sparse()
    values = data.index.get_level_values(0).values
    expected = pd.get_dummies(pd.Series(values, dtype='category'), drop_first=False)
    np.testing.assert_equal(self.grouping._dummies.toarray(), expected)
    if len(self.grouping.group_names) > 1:
        self.grouping.dummy_sparse(level=1)
        values = data.index.get_level_values(1).values
        expected = pd.get_dummies(pd.Series(values, dtype='category'), drop_first=False)
        np.testing.assert_equal(self.grouping._dummies.toarray(), expected)