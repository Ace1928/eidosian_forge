from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_duplicate_keys_in_function(self):
    self.flakes("\n            def f(thing):\n                pass\n            f({'yes': 1, 'yes': 2})\n            ", m.MultiValueRepeatedKeyLiteral, m.MultiValueRepeatedKeyLiteral)