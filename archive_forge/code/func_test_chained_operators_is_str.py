from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_chained_operators_is_str(self):
    self.flakes("\n        x = 5\n        if x is 'foo' < 4:\n            pass\n        ", IsLiteral)