import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_file_becomes_unversionable_bug_438569(self):
    self.requireFeature(features.OsFifoFeature)
    tree1 = self.make_branch_and_tree('1')
    self.build_tree(['1/a'])
    tree1.set_root_id(b'root-id')
    tree1.add(['a'], ids=[b'a-id'])
    tree2 = self.make_branch_and_tree('2')
    os.mkfifo('2/a')
    tree2.add(['a'], ['file'], [b'a-id'])
    try:
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
    except (KeyError,):
        raise tests.TestNotApplicable('Cannot represent a FIFO in this case %s' % self.id())
    try:
        self.do_iter_changes(tree1, tree2)
    except errors.BadFileKindError:
        pass