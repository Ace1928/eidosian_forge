import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
def test_add_equality_group_not_equivalent():
    eq = EqualsTester()
    with pytest.raises(AssertionError, match="can't be in the same"):
        eq.add_equality_group(1, 2)