import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
def test_add_equality_group_not_disjoint():
    eq = EqualsTester()
    eq.add_equality_group(1)
    with pytest.raises(AssertionError, match="can't be in different"):
        eq.add_equality_group(1)