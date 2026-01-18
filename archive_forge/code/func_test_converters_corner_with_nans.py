from io import StringIO
from dateutil.parser import parse
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_converters_corner_with_nans(all_parsers):
    parser = all_parsers
    data = 'id,score,days\n1,2,12\n2,2-5,\n3,,14+\n4,6-12,2'

    def convert_days(x):
        x = x.strip()
        if not x:
            return np.nan
        is_plus = x.endswith('+')
        if is_plus:
            x = int(x[:-1]) + 1
        else:
            x = int(x)
        return x

    def convert_days_sentinel(x):
        x = x.strip()
        if not x:
            return np.nan
        is_plus = x.endswith('+')
        if is_plus:
            x = int(x[:-1]) + 1
        else:
            x = int(x)
        return x

    def convert_score(x):
        x = x.strip()
        if not x:
            return np.nan
        if x.find('-') > 0:
            val_min, val_max = map(int, x.split('-'))
            val = 0.5 * (val_min + val_max)
        else:
            val = float(x)
        return val
    results = []
    for day_converter in [convert_days, convert_days_sentinel]:
        if parser.engine == 'pyarrow':
            msg = "The 'converters' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(StringIO(data), converters={'score': convert_score, 'days': day_converter}, na_values=['', None])
            continue
        result = parser.read_csv(StringIO(data), converters={'score': convert_score, 'days': day_converter}, na_values=['', None])
        assert pd.isna(result['days'][1])
        results.append(result)
    if parser.engine != 'pyarrow':
        tm.assert_frame_equal(results[0], results[1])