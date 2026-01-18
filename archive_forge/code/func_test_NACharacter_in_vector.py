import pytest
import math
import rpy2.rinterface as ri
def test_NACharacter_in_vector():
    na_str = ri.NA_Character
    x = ri.StrSexpVector(('ab', na_str, 'cd'))
    assert x[0] == 'ab'
    assert x.get_charsxp(1).rid == na_str.rid
    assert x[2] == 'cd'