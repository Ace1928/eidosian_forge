from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_merge_to_conflict_in_child_only(self):
    """When new_tags.merge_to(child.tags) conflicts with the child but not
        the master, a conflict is reported and the child receives the new tag.
        """
    master = self.make_branch('master')
    master.tags.set_tag('foo', b'rev-2')
    other = self.make_branch('other')
    other.tags.set_tag('foo', b'rev-1')
    child = self.make_branch('child')
    child.bind(master)
    child.update()
    master.tags.delete_tag('foo')
    tag_updates, tag_conflicts = other.tags.merge_to(child.tags)
    self.assertEqual(b'rev-2', child.tags.lookup_tag('foo'))
    self.assertEqual(b'rev-1', master.tags.lookup_tag('foo'))
    self.assertEqual([('foo', b'rev-1', b'rev-2')], list(tag_conflicts))
    self.assertEqual({'foo': b'rev-1'}, tag_updates)