import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewm_alpha_arg(series):
    s = series
    msg = 'Must pass one of comass, span, halflife, or alpha'
    with pytest.raises(ValueError, match=msg):
        s.ewm()
    msg = 'comass, span, halflife, and alpha are mutually exclusive'
    with pytest.raises(ValueError, match=msg):
        s.ewm(com=10.0, alpha=0.5)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=10.0, alpha=0.5)
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=10.0, alpha=0.5)