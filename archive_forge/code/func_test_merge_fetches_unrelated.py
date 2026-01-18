from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def test_merge_fetches_unrelated(self):
    """Merge brings across history from unrelated source"""
    wt1 = self.make_branch_and_tree('br1')
    br1 = wt1.branch
    wt1.commit(message='rev 1-1', rev_id=b'1-1')
    wt1.commit(message='rev 1-2', rev_id=b'1-2')
    wt2 = self.make_branch_and_tree('br2')
    br2 = wt2.branch
    wt2.commit(message='rev 2-1', rev_id=b'2-1')
    wt2.merge_from_branch(br1, from_revision=b'null:')
    self._check_revs_present(br2)