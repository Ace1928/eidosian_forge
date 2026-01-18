from breezy import multiwalker, revision
from breezy import tree as _mod_tree
from breezy.tests import TestCaseWithTransport
def lock_and_get_basis_and_root_id(self, tree):
    tree.lock_read()
    self.addCleanup(tree.unlock)
    basis_tree = tree.basis_tree()
    basis_tree.lock_read()
    self.addCleanup(basis_tree.unlock)
    root_id = tree.path2id('')
    return (basis_tree, root_id)