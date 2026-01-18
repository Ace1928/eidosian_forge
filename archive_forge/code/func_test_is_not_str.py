from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_is_not_str(self):
    self.flakes("\n        x = 'foo'\n        if x is not 'foo':\n            pass\n        ", IsLiteral)