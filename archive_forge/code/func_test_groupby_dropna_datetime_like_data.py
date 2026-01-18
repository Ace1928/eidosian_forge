import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.arm_slow
@pytest.mark.parametrize('datetime1, datetime2', [(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-01')), (pd.Timedelta('-2 days'), pd.Timedelta('-1 days')), (pd.Period('2020-01-01'), pd.Period('2020-02-01'))])
@pytest.mark.parametrize('dropna, values', [(True, [12, 3]), (False, [12, 3, 6])])
def test_groupby_dropna_datetime_like_data(dropna, values, datetime1, datetime2, unique_nulls_fixture, unique_nulls_fixture2):
    df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 6], 'dt': [datetime1, unique_nulls_fixture, datetime2, unique_nulls_fixture2, datetime1, datetime1]})
    if dropna:
        indexes = [datetime1, datetime2]
    else:
        indexes = [datetime1, datetime2, np.nan]
    grouped = df.groupby('dt', dropna=dropna).agg({'values': 'sum'})
    expected = pd.DataFrame({'values': values}, index=pd.Index(indexes, name='dt'))
    tm.assert_frame_equal(grouped, expected)