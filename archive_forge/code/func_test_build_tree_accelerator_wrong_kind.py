import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_build_tree_accelerator_wrong_kind(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    source = self.make_branch_and_tree('source')
    self.build_tree_contents([('source/file1', b'')])
    self.build_tree_contents([('source/file2', b'')])
    source.add(['file1', 'file2'], ids=[b'file1-id', b'file2-id'])
    source.commit('commit files')
    os.unlink('source/file2')
    self.build_tree_contents([('source/file2/', b'C')])
    os.unlink('source/file1')
    os.symlink('file2', 'source/file1')
    calls = []
    real_source_get_file = source.get_file

    def get_file(path):
        calls.append(path)
        return real_source_get_file(path)
    source.get_file = get_file
    target = self.make_branch_and_tree('target')
    revision_tree = source.basis_tree()
    revision_tree.lock_read()
    self.addCleanup(revision_tree.unlock)
    build_tree(revision_tree, target, source)
    self.assertEqual([], calls)
    target.lock_read()
    self.addCleanup(target.unlock)
    self.assertEqual([], list(target.iter_changes(revision_tree)))