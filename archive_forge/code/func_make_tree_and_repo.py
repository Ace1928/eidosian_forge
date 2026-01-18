from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def make_tree_and_repo(self):
    self.tree = self.make_branch_and_tree('tree', format='pack-0.92')
    self.repo = self.make_repository('rich-repo', format='rich-root-pack')
    self.repo.lock_write()
    self.addCleanup(self.repo.unlock)