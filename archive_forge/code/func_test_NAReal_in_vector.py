import pytest
import math
import rpy2.rinterface as ri
def test_NAReal_in_vector():
    na_float = ri.NA_Real
    x = ri.FloatSexpVector((1.1, na_float, 2.2))
    assert math.isnan(x[1])
    assert x[0] == 1.1
    assert x[2] == 2.2