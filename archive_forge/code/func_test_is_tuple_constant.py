from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_is_tuple_constant(self):
    self.flakes('            x = 5\n            if x is ():\n                pass\n        ', IsLiteral)