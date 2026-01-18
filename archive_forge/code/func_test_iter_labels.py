import pytest
from rpy2.rinterface import NA_Character
from rpy2 import robjects
@pytest.mark.parametrize('values,check_values', (('abaac', 'abaac'), (('abc', None, 'efg'), ('abc', NA_Character, 'efg'))))
def test_iter_labels(values, check_values):
    vec = robjects.FactorVector(robjects.StrVector(values))
    it = vec.iter_labels()
    for a, b in zip(check_values, it):
        assert a == b