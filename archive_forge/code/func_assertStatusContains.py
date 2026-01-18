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
def assertStatusContains(self, pattern, short=False):
    """Run status, and assert it contains the given pattern"""
    if short:
        result = self.run_bzr('status --short')[0]
    else:
        result = self.run_bzr('status')[0]
    self.assertContainsRe(result, pattern)