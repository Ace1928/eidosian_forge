from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from scipy import sparse
from statsmodels.tools.grouputils import (dummy_sparse, Grouping, Group,
from statsmodels.datasets import grunfeld, anes96
def test_init_api():
    grun_data = grunfeld.load_pandas().data
    multi_index_panel = grun_data.set_index(['firm', 'year']).index
    grouping = Grouping(multi_index_panel)
    np.testing.assert_array_equal(grouping.group_names, ['firm', 'year'])
    np.testing.assert_array_equal(grouping.index_shape, (11, 20))
    np.testing.assert_array_equal(grouping.labels, [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    grouping = Grouping(multi_index_panel, names=['firms', 'year'])
    np.testing.assert_array_equal(grouping.group_names, ['firms', 'year'])
    anes_data = anes96.load_pandas().data
    multi_index_groups = anes_data.set_index(['educ', 'income', 'TVnews']).index
    grouping = Grouping(multi_index_groups)
    np.testing.assert_array_equal(grouping.group_names, ['educ', 'income', 'TVnews'])
    np.testing.assert_array_equal(grouping.index_shape, (7, 24, 8))
    list_panel = multi_index_panel.tolist()
    grouping = Grouping(list_panel, names=['firms', 'year'])
    np.testing.assert_array_equal(grouping.group_names, ['firms', 'year'])
    np.testing.assert_array_equal(grouping.index_shape, (11, 20))
    list_groups = multi_index_groups.tolist()
    grouping = Grouping(list_groups, names=['educ', 'income', 'TVnews'])
    np.testing.assert_array_equal(grouping.group_names, ['educ', 'income', 'TVnews'])
    np.testing.assert_array_equal(grouping.index_shape, (7, 24, 8))
    index_group = multi_index_panel.get_level_values(0)
    grouping = Grouping(index_group)
    np.testing.assert_array_equal(grouping.group_names, ['firms'])
    np.testing.assert_array_equal(grouping.index_shape, (220,))
    list_group = multi_index_panel.get_level_values(0).tolist()
    grouping = Grouping(list_group)
    np.testing.assert_array_equal(grouping.group_names, ['group0'])
    np.testing.assert_array_equal(grouping.index_shape, 11 * 20)
    grouping = Grouping(list_groups)
    np.testing.assert_array_equal(grouping.group_names, ['group0', 'group1', 'group2'])