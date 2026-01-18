import pytest
import math
import rpy2.rinterface as ri
def test_NACharacter_str():
    na = ri.NA_Character
    assert str(na) == 'NA_character_'