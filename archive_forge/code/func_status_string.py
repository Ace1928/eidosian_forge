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
def status_string(self, wt, specific_files=None, revision=None, short=False, pending=True, verbose=False):
    uio = self.make_utf8_encoded_stringio()
    show_tree_status(wt, specific_files=specific_files, to_file=uio, revision=revision, short=short, show_pending=pending, verbose=verbose)
    return uio.getvalue().decode('utf-8')