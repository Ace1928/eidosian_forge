import pytest
from rpy2.rinterface import NA_Character
from rpy2 import robjects
def test_levels():
    vec = robjects.FactorVector(robjects.StrVector('abaabc'))
    assert len(vec.levels) == 3
    assert set(('a', 'b', 'c')) == set(tuple(vec.levels))