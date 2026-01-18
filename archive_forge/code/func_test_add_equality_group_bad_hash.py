import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
def test_add_equality_group_bad_hash():

    class KeyHash:

        def __init__(self, k, h):
            self._k = k
            self._h = h

        def __eq__(self, other):
            if not isinstance(other, KeyHash):
                return NotImplemented
            return self._k == other._k

        def __ne__(self, other):
            return not self == other

        def __hash__(self):
            return self._h
    eq = EqualsTester()
    eq.add_equality_group(KeyHash('a', 5), KeyHash('a', 5))
    eq.add_equality_group(KeyHash('b', 5))
    with pytest.raises(AssertionError, match='produced different hashes'):
        eq.add_equality_group(KeyHash('c', 2), KeyHash('c', 3))