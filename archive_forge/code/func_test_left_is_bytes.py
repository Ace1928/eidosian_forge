from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_left_is_bytes(self):
    self.flakes("\n        x = b'foo'\n        if b'foo' is x:\n            pass\n        ", IsLiteral)