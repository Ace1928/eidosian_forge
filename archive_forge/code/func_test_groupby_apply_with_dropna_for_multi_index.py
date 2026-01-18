import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('dropna, data, selected_data, levels', [pytest.param(False, {'groups': ['a', 'a', 'b', np.nan], 'values': [10, 10, 20, 30]}, {'values': [0, 1, 0, 0]}, ['a', 'b', np.nan], id='dropna_false_has_nan'), pytest.param(True, {'groups': ['a', 'a', 'b', np.nan], 'values': [10, 10, 20, 30]}, {'values': [0, 1, 0]}, None, id='dropna_true_has_nan'), pytest.param(False, {'groups': ['a', 'a', 'b', 'c'], 'values': [10, 10, 20, 30]}, {'values': [0, 1, 0, 0]}, None, id='dropna_false_no_nan'), pytest.param(True, {'groups': ['a', 'a', 'b', 'c'], 'values': [10, 10, 20, 30]}, {'values': [0, 1, 0, 0]}, None, id='dropna_true_no_nan')])
def test_groupby_apply_with_dropna_for_multi_index(dropna, data, selected_data, levels):
    df = pd.DataFrame(data)
    gb = df.groupby('groups', dropna=dropna)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = gb.apply(lambda grp: pd.DataFrame({'values': range(len(grp))}))
    mi_tuples = tuple(zip(data['groups'], selected_data['values']))
    mi = pd.MultiIndex.from_tuples(mi_tuples, names=['groups', None])
    if not dropna and levels:
        mi = mi.set_levels(levels, level='groups')
    expected = pd.DataFrame(selected_data, index=mi)
    tm.assert_frame_equal(result, expected)