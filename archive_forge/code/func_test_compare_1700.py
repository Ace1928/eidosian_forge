from pandas import Timestamp
def test_compare_1700(self):
    ts = Timestamp('1700-06-23')
    res = ts.to_julian_date()
    assert res == 2342145.5