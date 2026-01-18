import os
from breezy import osutils, tests, workingtree
def test_join_error(self):
    base_tree, sub_tree = self.make_trees()
    os.mkdir('tree/subtree2')
    osutils.rename('tree/subtree', 'tree/subtree2/subtree')
    self.run_bzr_error(('Cannot join .*subtree.  Parent directory is not versioned',), 'join tree/subtree2/subtree')
    self.run_bzr_error(('Not a branch:.*subtree2',), 'join tree/subtree2')