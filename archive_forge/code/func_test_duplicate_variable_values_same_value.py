from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_duplicate_variable_values_same_value(self):
    self.flakes('\n            a = 1\n            b = 1\n            {1: a, 1: b}\n            ', m.MultiValueRepeatedKeyLiteral, m.MultiValueRepeatedKeyLiteral)