from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_duplicate_keys_in_lambda(self):
    self.flakes('lambda x: {(0,1): 1, (0,1): 2}', m.MultiValueRepeatedKeyLiteral, m.MultiValueRepeatedKeyLiteral)