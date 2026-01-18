import pytest
import math
import rpy2.rinterface as ri
def test_NACharacter_repr():
    na = ri.NA_Character
    assert repr(na) == 'NA_character_'