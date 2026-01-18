import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx,labels,msg', [(period_range(start='2000', periods=20, freq='D'), Index(['4D', '8D'], dtype=object), "None of \\[Index\\(\\['4D', '8D'\\], dtype='object'\\)\\] are in the \\[index\\]"), (date_range(start='2000', periods=20, freq='D'), Index(['4D', '8D'], dtype=object), "None of \\[Index\\(\\['4D', '8D'\\], dtype='object'\\)\\] are in the \\[index\\]"), (pd.timedelta_range(start='1 day', periods=20), Index(['2000-01-04', '2000-01-08'], dtype=object), "None of \\[Index\\(\\['2000-01-04', '2000-01-08'\\], dtype='object'\\)\\] are in the \\[index\\]")])
def test_loc_with_list_of_strings_representing_datetimes_not_matched_type(self, idx, labels, msg):
    ser = Series(range(20), index=idx)
    df = DataFrame(range(20), index=idx)
    with pytest.raises(KeyError, match=msg):
        ser.loc[labels]
    with pytest.raises(KeyError, match=msg):
        ser[labels]
    with pytest.raises(KeyError, match=msg):
        df.loc[labels]