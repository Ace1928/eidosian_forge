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
def test_file_moves(self):
    """Test moves"""
    builder = MergeBuilder(getcwd())
    dir1 = builder.add_dir(builder.root(), 'dir1', file_id=b'1')
    dir2 = builder.add_dir(builder.root(), 'dir2', file_id=b'2')
    file1 = builder.add_file(dir1, 'file1', b'hello1', True, file_id=b'3')
    file2 = builder.add_file(dir1, 'file2', b'hello2', True, file_id=b'4')
    file3 = builder.add_file(dir1, 'file3', b'hello3', True, file_id=b'5')
    builder.change_parent(file1, other=b'2')
    builder.change_parent(file2, this=b'2')
    builder.change_parent(file3, base=b'2')
    builder.merge()
    builder.cleanup()
    builder = MergeBuilder(getcwd())
    dir1 = builder.add_dir(builder.root(), 'dir1', file_id=b'1')
    builder.add_dir(builder.root(), 'dir2', file_id=b'2')
    builder.add_dir(builder.root(), 'dir3', file_id=b'3')
    file1 = builder.add_file(dir1, 'file1', b'hello1', False, file_id=b'4')
    builder.change_parent(file1, other=b'2', this=b'3')
    conflicts = builder.merge()
    path2 = pathjoin('dir2', 'file1')
    path3 = pathjoin('dir3', 'file1')
    self.assertEqual(conflicts, [PathConflict(path3, path2, b'4')])
    builder.cleanup()