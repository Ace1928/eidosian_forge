import codecs
import sys
from io import BytesIO, StringIO
from os import chdir, mkdir, rmdir, unlink
import breezy.branch
from breezy.bzr import bzrdir, conflicts
from ... import errors, osutils, status
from ...osutils import pathjoin
from ...revisionspec import RevisionSpec
from ...status import show_tree_status
from ...workingtree import WorkingTree
from .. import TestCaseWithTransport, TestSkipped
def test_branch_status(self):
    """Test basic branch status"""
    wt = self.make_branch_and_tree('.')
    self.assertStatus([], wt)
    self.build_tree(['hello.c', 'bye.c'])
    self.assertStatus(['unknown:\n', '  bye.c\n', '  hello.c\n'], wt)
    self.assertStatus(['?   bye.c\n', '?   hello.c\n'], wt, short=True)
    wt.commit('create a parent to allow testing merge output')
    wt.add_parent_tree_id(b'pending@pending-0-0')
    self.assertStatus(['unknown:\n', '  bye.c\n', '  hello.c\n', 'pending merge tips: (use -v to see all merge revisions)\n', '  (ghost) pending@pending-0-0\n'], wt)
    self.assertStatus(['unknown:\n', '  bye.c\n', '  hello.c\n', 'pending merges:\n', '  (ghost) pending@pending-0-0\n'], wt, verbose=True)
    self.assertStatus(['?   bye.c\n', '?   hello.c\n', 'P   (ghost) pending@pending-0-0\n'], wt, short=True)
    self.assertStatus(['unknown:\n', '  bye.c\n', '  hello.c\n'], wt, pending=False)
    self.assertStatus(['?   bye.c\n', '?   hello.c\n'], wt, short=True, pending=False)