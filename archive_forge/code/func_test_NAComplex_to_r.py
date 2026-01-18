import pytest
import math
import rpy2.rinterface as ri
def test_NAComplex_to_r():
    na_complex = ri.NA_Complex
    assert ri.baseenv['is.na'](na_complex)[0]