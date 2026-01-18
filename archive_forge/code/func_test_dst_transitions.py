from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
def test_dst_transitions(setup_path):
    with ensure_clean_store(setup_path) as store:
        times = date_range('2013-10-26 23:00', '2013-10-27 01:00', tz='Europe/London', freq='h', ambiguous='infer')
        times = times._with_freq(None)
        for i in [times, times + pd.Timedelta('10min')]:
            _maybe_remove(store, 'df')
            df = DataFrame({'A': range(len(i)), 'B': i}, index=i)
            store.append('df', df)
            result = store.select('df')
            tm.assert_frame_equal(result, df)