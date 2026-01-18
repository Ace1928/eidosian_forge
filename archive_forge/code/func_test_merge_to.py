from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def test_merge_to(self):
    other_tags = MemoryTags({})
    other_tags.set_tag('tag-1', b'x')
    self.tags.set_tag('tag-2', b'y')
    other_tags.merge_to(self.tags)
    self.assertEqual(b'x', self.tags.lookup_tag('tag-1'))
    self.assertEqual(b'y', self.tags.lookup_tag('tag-2'))
    self.assertRaises(errors.NoSuchTag, other_tags.lookup_tag, 'tag-2')
    other_tags.set_tag('tag-2', b'z')
    updates, conflicts = other_tags.merge_to(self.tags)
    self.assertEqual({}, updates)
    self.assertEqual(list(conflicts), [('tag-2', b'z', b'y')])
    self.assertEqual(b'y', self.tags.lookup_tag('tag-2'))
    updates, conflicts = other_tags.merge_to(self.tags, overwrite=True)
    self.assertEqual(list(conflicts), [])
    self.assertEqual({'tag-2': b'z'}, updates)
    self.assertEqual(b'z', self.tags.lookup_tag('tag-2'))