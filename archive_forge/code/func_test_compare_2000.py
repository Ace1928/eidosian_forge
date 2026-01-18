from pandas import Timestamp
def test_compare_2000(self):
    ts = Timestamp('2000-04-12')
    res = ts.to_julian_date()
    assert res == 2451646.5