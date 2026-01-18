from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def test_merge_fetches(self):
    """Merge brings across history from source"""
    wt1 = self.make_branch_and_tree('br1')
    br1 = wt1.branch
    wt1.commit(message='rev 1-1', rev_id=b'1-1')
    dir_2 = br1.controldir.sprout('br2')
    br2 = dir_2.open_branch()
    wt1.commit(message='rev 1-2', rev_id=b'1-2')
    wt2 = dir_2.open_workingtree()
    wt2.commit(message='rev 2-1', rev_id=b'2-1')
    wt2.merge_from_branch(br1)
    self._check_revs_present(br2)