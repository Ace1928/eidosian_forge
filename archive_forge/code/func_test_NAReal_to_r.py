import pytest
import math
import rpy2.rinterface as ri
def test_NAReal_to_r():
    na_real = ri.NA_Real
    assert ri.baseenv['is.na'](na_real)[0]