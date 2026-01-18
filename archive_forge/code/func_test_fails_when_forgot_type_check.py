import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
def test_fails_when_forgot_type_check():
    eq = EqualsTester()

    class NoTypeCheckEqualImplementation:

        def __init__(self):
            self.x = 1

        def __eq__(self, other):
            return self.x == other.x

        def __ne__(self, other):
            return not self == other

        def __hash__(self):
            return hash(self.x)
    with pytest.raises(AttributeError, match="has no attribute 'x'"):
        eq.add_equality_group(NoTypeCheckEqualImplementation())