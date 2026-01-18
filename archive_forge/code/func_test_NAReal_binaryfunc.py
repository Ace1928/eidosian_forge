import pytest
import math
import rpy2.rinterface as ri
def test_NAReal_binaryfunc():
    na_real = ri.NA_Real
    assert math.isnan(na_real + 2.0)