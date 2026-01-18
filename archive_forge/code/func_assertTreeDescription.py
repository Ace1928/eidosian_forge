import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def assertTreeDescription(self, format):
    """Assert a tree's format description matches expectations"""
    self.make_branch_and_tree('%s_tree' % format, format=format)
    tree = workingtree.WorkingTree.open('%s_tree' % format)
    self.assertEqual(format, info.describe_format(tree.controldir, tree.branch.repository, tree.branch, tree))