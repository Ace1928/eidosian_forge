from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_is_not_true(self):
    self.flakes('\n        x = True\n        if x is not True:\n            pass\n        ')