from datetime import datetime
import pytz
from pandas import DataFrame
import pandas._testing as tm
def test_assign_columns(self, float_frame):
    float_frame['hi'] = 'there'
    df = float_frame.copy()
    df.columns = ['foo', 'bar', 'baz', 'quux', 'foo2']
    tm.assert_series_equal(float_frame['C'], df['baz'], check_names=False)
    tm.assert_series_equal(float_frame['hi'], df['foo2'], check_names=False)