from breezy import errors
from breezy.bzr.inventorytree import InventoryTree
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def assertExpectedIds(self, ids, tree, paths, trees=None, require_versioned=True):
    """Run paths2ids for tree, and check the result."""
    tree.lock_read()
    if trees:
        for t in trees:
            t.lock_read()
        result = tree.paths2ids(paths, trees, require_versioned=require_versioned)
        for t in trees:
            t.unlock()
    else:
        result = tree.paths2ids(paths, require_versioned=require_versioned)
    self.assertEqual(set(ids), result)
    tree.unlock()