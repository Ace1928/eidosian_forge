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
def test_kind_change_plain(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['file'])
    tree.add('file')
    tree.commit('added file')
    unlink('file')
    self.build_tree(['file/'])
    self.assertStatusContains('kind changed:\n  file \\(file => directory\\)')
    tree.rename_one('file', 'directory')
    self.assertStatusContains('renamed:\n  file => directory/\nmodified:\n  directory/\n')
    rmdir('directory')
    self.assertStatusContains('removed:\n  file\n')