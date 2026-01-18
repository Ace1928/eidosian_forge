from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_cached_tag_dict_not_accidentally_mutable(self):
    """When there's a cached version of the tags, b.tags.get_tag_dict
        returns a copy of the cached data so that callers cannot accidentally
        corrupt the cache.
        """
    b, [rev1, rev2, rev3] = self.make_branch_with_revision_tuple('b', 3)
    b.tags.set_tag('one', rev1)
    self.addCleanup(b.lock_read().unlock)
    tags_dict = b.tags.get_tag_dict()
    tags_dict['two'] = rev2
    tags_dict = b.tags.get_tag_dict()
    tags_dict['three'] = rev3
    self.assertEqual({'one': rev1}, b.tags.get_tag_dict())