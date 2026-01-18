import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_build_tree_accelerator_tree_moved(self):
    source = self.make_branch_and_tree('source')
    self.build_tree_contents([('source/file1', b'A')])
    source.add(['file1'], ids=[b'file1-id'])
    source.commit('commit files')
    source.rename_one('file1', 'file2')
    source.lock_read()
    self.addCleanup(source.unlock)
    target = self.make_branch_and_tree('target')
    revision_tree = source.basis_tree()
    revision_tree.lock_read()
    self.addCleanup(revision_tree.unlock)
    build_tree(revision_tree, target, source)
    target.lock_read()
    self.addCleanup(target.unlock)
    self.assertEqual([], list(target.iter_changes(revision_tree)))