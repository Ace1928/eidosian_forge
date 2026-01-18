from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_duplicate_keys_tuples_int_and_float(self):
    self.flakes('{(0,1): 1, (0,1.0): 2}', m.MultiValueRepeatedKeyLiteral, m.MultiValueRepeatedKeyLiteral)