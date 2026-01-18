import pytest
import math
import rpy2.rinterface as ri
def test_NAReal_repr():
    na_float = ri.NA_Real
    assert repr(na_float) == 'NA_real_'