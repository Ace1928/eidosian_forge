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
def test_merge_one_renamed(self):
    builder = MergeBuilder(getcwd())
    name1 = builder.add_file(builder.root(), 'name1', b'text1a', False, file_id=b'1')
    builder.change_name(name1, this='name2')
    builder.change_contents(name1, other=b'text2')
    builder.merge(interesting_files=['name2'])
    self.assertEqual(b'text2', builder.this.get_file('name2').read())
    builder.cleanup()