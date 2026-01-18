import pytest
import math
import rpy2.rinterface as ri
def test_NAInteger_to_r():
    na_int = ri.NA_Integer
    assert ri.baseenv['is.na'](na_int)[0]