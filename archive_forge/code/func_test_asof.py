from datetime import timedelta
from pandas import (
import pandas._testing as tm
def test_asof(self):
    index = tm.makeDateIndex(100)
    dt = index[0]
    assert index.asof(dt) == dt
    assert isna(index.asof(dt - timedelta(1)))
    dt = index[-1]
    assert index.asof(dt + timedelta(1)) == dt
    dt = index[0].to_pydatetime()
    assert isinstance(index.asof(dt), Timestamp)