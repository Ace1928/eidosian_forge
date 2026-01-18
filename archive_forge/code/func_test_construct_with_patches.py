from testtools import TestCase
from testtools.matchers import MatchesException, Raises
from testtools.monkey import MonkeyPatcher, patch
def test_construct_with_patches(self):
    patcher = MonkeyPatcher((self.test_object, 'foo', 'haha'), (self.test_object, 'bar', 'hehe'))
    patcher.patch()
    self.assertEqual('haha', self.test_object.foo)
    self.assertEqual('hehe', self.test_object.bar)
    self.assertEqual(self.original_object.baz, self.test_object.baz)