from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def test_merge_fetches_file_history(self):
    """Merge brings across file histories"""
    br2 = Branch.open('br2')
    br1 = Branch.open('br1')
    wt2 = WorkingTree.open('br2').merge_from_branch(br1)
    br2.lock_read()
    self.addCleanup(br2.unlock)
    for rev_id, text in [(b'1-2', b'original from 1\n'), (b'1-3', b'agreement\n'), (b'2-1', b'contents in 2\n'), (b'2-2', b'agreement\n')]:
        self.assertEqualDiff(br2.repository.revision_tree(rev_id).get_file_text('file'), text)