from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_is_false(self):
    self.flakes('\n        x = False\n        if x is False:\n            pass\n        ')