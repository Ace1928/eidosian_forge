from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def test_get_reverse_tag_dict(self):
    self.assertEqual(self.tags.get_reverse_tag_dict(), {})