import fractions
import pytest
import cirq
def test_add_ordering_group_correct():
    ot = cirq.testing.OrderTester()
    ot.add_ascending(-4, 0)
    ot.add_ascending(1, 2)
    ot.add_ascending_equivalence_group(fractions.Fraction(6, 2), fractions.Fraction(12, 4), 3, 3.0)
    ot.add_ascending_equivalence_group(float('inf'), float('inf'))