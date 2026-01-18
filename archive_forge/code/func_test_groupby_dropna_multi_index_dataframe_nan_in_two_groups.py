import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('dropna, tuples, outputs', [(True, [['A', 'B'], ['B', 'A']], {'c': [12.0, 123.23], 'd': [12.0, 123.0], 'e': [12.0, 1.0]}), (False, [['A', 'B'], ['A', np.nan], ['B', 'A'], [np.nan, 'B']], {'c': [12.0, 13.3, 123.23, 1.0], 'd': [12.0, 234.0, 123.0, 1.0], 'e': [12.0, 13.0, 1.0, 1.0]})])
def test_groupby_dropna_multi_index_dataframe_nan_in_two_groups(dropna, tuples, outputs, nulls_fixture, nulls_fixture2):
    df_list = [['A', 'B', 12, 12, 12], ['A', nulls_fixture, 12.3, 233.0, 12], ['B', 'A', 123.23, 123, 1], [nulls_fixture2, 'B', 1, 1, 1.0], ['A', nulls_fixture2, 1, 1, 1.0]]
    df = pd.DataFrame(df_list, columns=['a', 'b', 'c', 'd', 'e'])
    grouped = df.groupby(['a', 'b'], dropna=dropna).sum()
    mi = pd.MultiIndex.from_tuples(tuples, names=list('ab'))
    if not dropna:
        mi = mi.set_levels([['A', 'B', np.nan], ['A', 'B', np.nan]])
    expected = pd.DataFrame(outputs, index=mi)
    tm.assert_frame_equal(grouped, expected)