import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def test_merge_specific_file(self):
    this_tree = self.make_branch_and_tree('this')
    this_tree.lock_write()
    self.addCleanup(this_tree.unlock)
    self.build_tree_contents([('this/file1', b'a\nb\n'), ('this/file2', b'a\nb\n')])
    this_tree.add(['file1', 'file2'])
    this_tree.commit('Added files')
    other_tree = this_tree.controldir.sprout('other').open_workingtree()
    self.build_tree_contents([('other/file1', b'a\nb\nc\n'), ('other/file2', b'a\nb\nc\n')])
    other_tree.commit('modified both')
    self.build_tree_contents([('this/file1', b'd\na\nb\n'), ('this/file2', b'd\na\nb\n')])
    this_tree.commit('modified both')
    self.do_merge(this_tree, other_tree, interesting_files=['file1'])
    self.assertFileEqual(b'd\na\nb\nc\n', 'this/file1')
    self.assertFileEqual(b'd\na\nb\n', 'this/file2')