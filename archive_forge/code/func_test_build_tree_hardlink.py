import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_build_tree_hardlink(self):
    self.requireFeature(features.HardlinkFeature(self.test_dir))
    source = self.create_ab_tree()
    target = self.make_branch_and_tree('target')
    revision_tree = source.basis_tree()
    revision_tree.lock_read()
    self.addCleanup(revision_tree.unlock)
    build_tree(revision_tree, target, source, hardlink=True)
    target.lock_read()
    self.addCleanup(target.unlock)
    self.assertEqual([], list(target.iter_changes(revision_tree)))
    source_stat = os.stat('source/file1')
    target_stat = os.stat('target/file1')
    self.assertEqual(source_stat, target_stat)
    target2 = self.make_branch_and_tree('target2')
    build_tree(revision_tree, target2, source, hardlink=False)
    target2.lock_read()
    self.addCleanup(target2.unlock)
    self.assertEqual([], list(target2.iter_changes(revision_tree)))
    source_stat = os.stat('source/file1')
    target2_stat = os.stat('target2/file1')
    self.assertNotEqual(source_stat, target2_stat)