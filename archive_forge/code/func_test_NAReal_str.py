import pytest
import math
import rpy2.rinterface as ri
def test_NAReal_str():
    na_float = ri.NA_Real
    assert str(na_float) == 'NA_real_'