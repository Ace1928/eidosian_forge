from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def make_repository_tree(self):
    self.build_tree(['root/'])
    self.make_repository('root', shared=True)
    tree = self.make_branch_and_tree('root/tree')
    reconfigure.Reconfigure.to_use_shared(tree.controldir).apply()
    return workingtree.WorkingTree.open('root/tree')