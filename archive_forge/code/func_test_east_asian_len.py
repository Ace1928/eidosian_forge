import string
import pandas._config.config as cf
from pandas.io.formats import printing
def test_east_asian_len(self):
    adj = printing._EastAsianTextAdjustment()
    assert adj.len('abc') == 3
    assert adj.len('abc') == 3
    assert adj.len('パンダ') == 6
    assert adj.len('ﾊﾟﾝﾀﾞ') == 5
    assert adj.len('パンダpanda') == 11
    assert adj.len('ﾊﾟﾝﾀﾞpanda') == 10