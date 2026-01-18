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
def test_no_passive_add(self):
    builder = MergeBuilder(getcwd())
    name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
    builder.remove_file(name1, this=True)
    builder.merge()
    builder.cleanup()