from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_is_true(self):
    self.flakes('\n        x = True\n        if x is True:\n            pass\n        ')