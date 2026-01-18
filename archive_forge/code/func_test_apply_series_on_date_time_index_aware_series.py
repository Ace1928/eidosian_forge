import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('dti,exp', [(Series([1, 2], index=pd.DatetimeIndex([0, 31536000000])), DataFrame(np.repeat([[1, 2]], 2, axis=0), dtype='int64')), (Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts'), DataFrame(np.repeat([[1, 2]], 10, axis=0), dtype='int64'))])
@pytest.mark.parametrize('aware', [True, False])
def test_apply_series_on_date_time_index_aware_series(dti, exp, aware):
    if aware:
        index = dti.tz_localize('UTC').index
    else:
        index = dti.index
    result = Series(index).apply(lambda x: Series([1, 2]))
    tm.assert_frame_equal(result, exp)