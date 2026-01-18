import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
def test_fails_when_not_reflexive():
    eq = EqualsTester()

    class NotReflexiveImplementation:

        def __init__(self):
            self.x = 1

        def __eq__(self, other):
            if other is not self:
                return NotImplemented
            return False

        def __ne__(self, other):
            return not self == other
    with pytest.raises(AssertionError, match='equal to itself'):
        eq.add_equality_group(NotReflexiveImplementation())