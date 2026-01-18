from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_is_tuple_constant_containing_constants(self):
    self.flakes("            x = 5\n            if x is (1, '2', True, (1.5, ())):\n                pass\n        ", IsLiteral)