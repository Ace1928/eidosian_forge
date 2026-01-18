from breezy import multiwalker, revision
from breezy import tree as _mod_tree
from breezy.tests import TestCaseWithTransport
def test_master_renamed_to_earlier(self):
    """The record is still present, it just shows up early."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a', 'tree/c', 'tree/d'])
    tree.add(['a', 'c', 'd'], ids=[b'a-id', b'c-id', b'd-id'])
    tree.commit('first', rev_id=b'first-rev-id')
    tree.rename_one('d', 'b')
    basis_tree, root_id = self.lock_and_get_basis_and_root_id(tree)
    walker = multiwalker.MultiWalker(tree, [basis_tree])
    iterator = walker.iter_all()
    self.assertWalkerNext('', root_id, True, [''], iterator)
    self.assertWalkerNext('a', b'a-id', True, ['a'], iterator)
    self.assertWalkerNext('b', b'd-id', True, ['d'], iterator)
    self.assertWalkerNext('c', b'c-id', True, ['c'], iterator)
    self.assertRaises(StopIteration, next, iterator)