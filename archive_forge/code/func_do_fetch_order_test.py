from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def do_fetch_order_test(self, first, second):
    """Test that fetch works no matter what the set order of revision is.

        This test depends on the order of items in a set, which is
        implementation-dependant, so we test A, B and then B, A.
        """
    self.make_tree_and_repo()
    self.tree.commit('Commit 1', rev_id=first)
    self.tree.commit('Commit 2', rev_id=second)
    self.repo.fetch(self.tree.branch.repository, second)