from pandas import Series
import pandas._testing as tm
def test_combine_scalar(self):
    ser = Series([i * 10 for i in range(5)])
    result = ser.combine(3, lambda x, y: x + y)
    expected = Series([i * 10 + 3 for i in range(5)])
    tm.assert_series_equal(result, expected)
    result = ser.combine(22, lambda x, y: min(x, y))
    expected = Series([min(i * 10, 22) for i in range(5)])
    tm.assert_series_equal(result, expected)