import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
def test_add_equality_group_correct():
    eq = EqualsTester()
    eq.add_equality_group(fractions.Fraction(1, 1))
    eq.add_equality_group(fractions.Fraction(1, 2), fractions.Fraction(2, 4))
    eq.add_equality_group(fractions.Fraction(2, 3), fractions.Fraction(12, 18), fractions.Fraction(14, 21))
    eq.add_equality_group(2, 2.0, fractions.Fraction(2, 1))
    eq.add_equality_group([1, 2, 3], [1, 2, 3])
    eq.add_equality_group({'b': 3, 'a': 2}, {'a': 2, 'b': 3})
    eq.add_equality_group('unrelated')