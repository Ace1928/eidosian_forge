from testtools.matchers import *
from . import CapturedCall, TestCase, TestCaseWithTransport
from .matchers import *
def test_no_dirs(self):
    t = self.make_branch_and_tree('.')
    t.has_versioned_directories = lambda: False
    self.build_tree(['a', 'b/', 'b/c'])
    t.add(['a', 'b', 'b/c'], ids=[b'a-id', b'b-id', b'c-id'])
    self.assertIs(None, HasLayout(['', 'a', 'b/', 'b/c']).match(t))
    self.assertIs(None, HasLayout(['', 'a', 'b/', 'b/c', 'd/']).match(t))
    mismatch = HasLayout(['', 'a', 'd/']).match(t)
    self.assertIsNot(None, mismatch)
    self.assertEqual({"['', 'a', 'b/', 'b/c']", "['', 'a']"}, set(mismatch.describe().split(' != ')))