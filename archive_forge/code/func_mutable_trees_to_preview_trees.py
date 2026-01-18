import breezy
from breezy import revisiontree, tests
from breezy.bzr import inventorytree
from breezy.bzr.workingtree_3 import WorkingTreeFormat3
from breezy.bzr.workingtree_4 import WorkingTreeFormat4
from breezy.tests import default_transport, multiply_tests
from breezy.tests.per_tree import (TestCaseWithTree, return_parameter,
from breezy.tree import InterTree
def mutable_trees_to_preview_trees(test_case, source, target):
    preview = target.preview_transform()
    test_case.addCleanup(preview.finalize)
    return (source, preview.get_preview_tree())