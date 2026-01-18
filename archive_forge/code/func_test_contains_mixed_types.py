import pytest
from pandas import (
@pytest.mark.parametrize('type1', [(0, 1), (Timestamp(2000, 1, 1, 0), Timestamp(2000, 1, 1, 1)), (Timedelta('0h'), Timedelta('1h'))])
@pytest.mark.parametrize('type2', [(0, 1), (Timestamp(2000, 1, 1, 0), Timestamp(2000, 1, 1, 1)), (Timedelta('0h'), Timedelta('1h'))])
def test_contains_mixed_types(self, type1, type2):
    interval1 = Interval(*type1)
    interval2 = Interval(*type2)
    if type1 == type2:
        assert interval1 in interval2
    else:
        msg = "^'<=' not supported between instances of"
        with pytest.raises(TypeError, match=msg):
            interval1 in interval2