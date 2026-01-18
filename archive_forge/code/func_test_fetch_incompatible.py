from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def test_fetch_incompatible(self):
    knit_tree = self.make_branch_and_tree('knit', format='knit')
    knit3_tree = self.make_branch_and_tree('knit3', format='dirstate-with-subtree')
    knit3_tree.commit('blah')
    e = self.assertRaises(errors.IncompatibleRepositories, knit_tree.branch.fetch, knit3_tree.branch)
    self.assertContainsRe(str(e), '(?m).*/knit.*\\nis not compatible with\\n.*/knit3/.*\\ndifferent rich-root support')