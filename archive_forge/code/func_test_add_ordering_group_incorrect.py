import fractions
import pytest
import cirq
def test_add_ordering_group_incorrect():
    ot = cirq.testing.OrderTester()
    ot.add_ascending(0)
    with pytest.raises(AssertionError):
        ot.add_ascending_equivalence_group(0, 0)
    ot.add_ascending(1, 2)
    with pytest.raises(AssertionError):
        ot.add_ascending(20, 20)
    with pytest.raises(AssertionError):
        ot.add_ascending(1, 3)
    with pytest.raises(AssertionError):
        ot.add_ascending(6, 6)
    with pytest.raises(AssertionError):
        ot.add_ascending(99, 10)
    with pytest.raises(AssertionError):
        ot.add_ascending(0)