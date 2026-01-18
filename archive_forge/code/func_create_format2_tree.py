import os
from ... import conflicts, errors
from ...bzr.conflicts import ContentsConflict, TextConflict
from ...tests import TestCaseWithTransport
from .bzrdir import BzrDirFormat6
def create_format2_tree(self, url):
    return self.make_branch_and_tree(url, format=BzrDirFormat6())