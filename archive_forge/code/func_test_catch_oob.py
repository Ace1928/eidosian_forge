import pytest
from pandas.errors import (
import pandas as pd
def test_catch_oob():
    from pandas import errors
    msg = "Cannot cast 1500-01-01 00:00:00 to unit='ns' without overflow"
    with pytest.raises(errors.OutOfBoundsDatetime, match=msg):
        pd.Timestamp('15000101').as_unit('ns')