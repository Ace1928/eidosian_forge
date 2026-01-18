import pytest
import math
import rpy2.rinterface as ri
def test_NAInteger_in_vector():
    na_int = ri.NA_Integer
    x = ri.IntSexpVector((1, na_int, 2))
    assert x[1] is na_int
    assert x[0] == 1
    assert x[2] == 2