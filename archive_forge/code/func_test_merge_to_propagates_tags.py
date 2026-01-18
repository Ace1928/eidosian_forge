from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_merge_to_propagates_tags(self):
    """merge_to(child) also merges tags to the master."""
    master = self.make_branch('master')
    other = self.make_branch('other')
    other.tags.set_tag('foo', b'rev-1')
    child = self.make_branch('child')
    child.bind(master)
    child.update()
    other.tags.merge_to(child.tags)
    self.assertEqual(b'rev-1', child.tags.lookup_tag('foo'))
    self.assertEqual(b'rev-1', master.tags.lookup_tag('foo'))