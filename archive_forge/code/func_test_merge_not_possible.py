from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def test_merge_not_possible(self):
    old_branch = self.make_knit_branch('old')
    new_branch = self.make_branch_supporting_tags('new')
    self.assertFalse(old_branch.supports_tags(), '%s is expected to not support tags but does' % old_branch)
    self.assertTrue(new_branch.supports_tags(), '%s is expected to support tags but does not' % new_branch)
    old_branch.tags.merge_to(new_branch.tags)
    new_branch.tags.merge_to(old_branch.tags)
    new_branch.tags.set_tag('⁀tag', b'revid')
    old_branch.tags.merge_to(new_branch.tags)
    self.assertRaises(errors.TagsNotSupported, new_branch.tags.merge_to, old_branch.tags)