from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def test_merge_to_with_selector(self):
    a = self.make_branch_supporting_tags('a')
    b = self.make_branch_supporting_tags('b')
    a.tags.set_tag('tag-1', b'x')
    a.tags.set_tag('tag-2', b'y')
    updates, conflicts = a.tags.merge_to(b.tags, selector=lambda x: x == 'tag-1')
    self.assertEqual(list(conflicts), [])
    self.assertEqual({'tag-1': b'x'}, updates)
    self.assertRaises(errors.NoSuchTag, b.tags.lookup_tag, 'tag-2')