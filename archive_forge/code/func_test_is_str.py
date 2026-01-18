from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_is_str(self):
    self.flakes("\n        x = 'foo'\n        if x is 'foo':\n            pass\n        ", IsLiteral)