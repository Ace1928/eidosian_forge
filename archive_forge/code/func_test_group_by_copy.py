import numpy as np
import pandas as pd
import pandas._testing as tm
def test_group_by_copy():
    df = pd.DataFrame({'name': ['Alice', 'Bob', 'Carl'], 'age': [20, 21, 20]}).set_index('name')
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        grp_by_same_value = df.groupby(['age'], group_keys=False).apply(lambda group: group)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        grp_by_copy = df.groupby(['age'], group_keys=False).apply(lambda group: group.copy())
    tm.assert_frame_equal(grp_by_same_value, grp_by_copy)