from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_left_is_unicode(self):
    self.flakes("\n        x = u'foo'\n        if u'foo' is x:\n            pass\n        ", IsLiteral)