import pytest
from rpy2.rinterface import NA_Character
from rpy2 import robjects
def test_isordered():
    vec = robjects.FactorVector(robjects.StrVector('abaabc'))
    assert vec.isordered is False