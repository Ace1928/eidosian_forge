import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewma_halflife_arg(series):
    A = series.ewm(com=13.932726172912965).mean()
    B = series.ewm(halflife=10.0).mean()
    tm.assert_almost_equal(A, B)
    msg = 'comass, span, halflife, and alpha are mutually exclusive'
    with pytest.raises(ValueError, match=msg):
        series.ewm(span=20, halflife=50)
    with pytest.raises(ValueError, match=msg):
        series.ewm(com=9.5, halflife=50)
    with pytest.raises(ValueError, match=msg):
        series.ewm(com=9.5, span=20, halflife=50)
    msg = 'Must pass one of comass, span, halflife, or alpha'
    with pytest.raises(ValueError, match=msg):
        series.ewm()