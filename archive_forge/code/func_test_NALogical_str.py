import pytest
import math
import rpy2.rinterface as ri
def test_NALogical_str():
    na = ri.NA_Logical
    assert str(na) == 'NA'