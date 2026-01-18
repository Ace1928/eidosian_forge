from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_left_is_not_str(self):
    self.flakes("\n        x = 'foo'\n        if 'foo' is not x:\n            pass\n        ", IsLiteral)