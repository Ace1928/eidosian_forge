from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_list_tags_revision_filtering(self):
    tree1 = self.make_branch_and_tree('.')
    tree1.commit(allow_pointless=True, message='revision 1', rev_id=b'revid-1')
    tree1.commit(allow_pointless=True, message='revision 2', rev_id=b'revid-2')
    tree1.commit(allow_pointless=True, message='revision 3', rev_id=b'revid-3')
    tree1.commit(allow_pointless=True, message='revision 4', rev_id=b'revid-4')
    b1 = tree1.branch
    b1.tags.set_tag('tag 1', b'revid-1')
    b1.tags.set_tag('tag 2', b'revid-2')
    b1.tags.set_tag('tag 3', b'revid-3')
    b1.tags.set_tag('tag 4', b'revid-4')
    self._check_tag_filter('', (1, 2, 3, 4))
    self._check_tag_filter('-r ..', (1, 2, 3, 4))
    self._check_tag_filter('-r ..2', (1, 2))
    self._check_tag_filter('-r 2..', (2, 3, 4))
    self._check_tag_filter('-r 2..3', (2, 3))
    self._check_tag_filter('-r 3..2', ())
    self.run_bzr_error(args='tags -r 123', error_regexes=["brz: ERROR: Requested revision: '123' does not exist in branch:"])
    self.run_bzr_error(args='tags -r ..123', error_regexes=["brz: ERROR: Requested revision: '123' does not exist in branch:"])
    self.run_bzr_error(args='tags -r 123.123', error_regexes=["brz: ERROR: Requested revision: '123.123' does not exist in branch:"])