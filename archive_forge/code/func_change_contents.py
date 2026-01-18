import os
import sys
import tempfile
import breezy
from .. import controldir, errors, merge_directive, osutils
from ..bzr import generate_ids
from ..bzr.conflicts import ContentsConflict, PathConflict, TextConflict
from ..merge import Diff3Merger, Merge3Merger, Merger, WeaveMerger
from ..osutils import getcwd, pathjoin
from ..workingtree import WorkingTree
from . import TestCaseWithTransport, TestSkipped, features
def change_contents(self, trans_id, base=None, this=None, other=None):
    for trans_id, (contents, tt) in zip(trans_id, self.selected_transforms(this, base, other)):
        tt.cancel_creation(trans_id)
        tt.create_file([contents], trans_id)