import pytest
from rpy2.rinterface import NA_Character
from rpy2 import robjects
def test_factor_repr():
    vec = robjects.vectors.FactorVector(('abc', 'def', 'ghi'))
    s = repr(vec)
    assert s.endswith('[abc, def, ghi]')