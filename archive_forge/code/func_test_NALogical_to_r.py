import pytest
import math
import rpy2.rinterface as ri
def test_NALogical_to_r():
    na_lgl = ri.NA_Logical
    assert ri.baseenv['is.na'](na_lgl)[0] is True