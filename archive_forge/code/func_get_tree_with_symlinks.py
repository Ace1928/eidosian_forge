import os
from breezy import osutils, tests
from breezy.tests import features, per_tree
def get_tree_with_symlinks(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    tree = self.make_branch_and_tree('tree')
    os.symlink('foo', 'tree/link')
    os.symlink('../bar', 'tree/rel_link')
    os.symlink('/baz/bing', 'tree/abs_link')
    tree.add(['link', 'rel_link', 'abs_link'])
    return self._convert_tree(tree)