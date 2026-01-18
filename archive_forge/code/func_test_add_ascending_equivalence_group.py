import fractions
import pytest
import cirq
def test_add_ascending_equivalence_group():
    ot = cirq.testing.OrderTester()
    with pytest.raises(AssertionError, match='Expected X=1 to equal Y=3'):
        ot.add_ascending_equivalence_group(1, 3)
    ot.add_ascending_equivalence_group(2)
    ot.add_ascending_equivalence_group(4)
    with pytest.raises(AssertionError, match='Expected X=4 to be less than Y=3'):
        ot.add_ascending_equivalence_group(3)
    ot.add_ascending_equivalence_group(5)