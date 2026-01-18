from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def test_commit_trivial(self):
    """Smoke test for commit on a MemoryTree.

        Becamse of commits design and layering, if this works, all commit
        logic should work quite reliably.
        """
    branch = self.make_branch('branch')
    tree = branch.create_memorytree()
    with tree.lock_write():
        tree.add(['', 'foo'], kinds=['directory', 'file'])
        tree.put_file_bytes_non_atomic('foo', b'barshoom')
        revision_id = tree.commit('message baby')
        self.assertEqual([revision_id], tree.get_parent_ids())
    revtree = tree.branch.repository.revision_tree(revision_id)
    with revtree.lock_read():
        self.assertEqual(b'barshoom', revtree.get_file('foo').read())