from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_is_bytes(self):
    self.flakes("\n        x = b'foo'\n        if x is b'foo':\n            pass\n        ", IsLiteral)