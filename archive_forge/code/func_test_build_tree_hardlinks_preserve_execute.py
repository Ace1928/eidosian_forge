import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_build_tree_hardlinks_preserve_execute(self):
    self.requireFeature(features.HardlinkFeature(self.test_dir))
    source = self.create_ab_tree()
    tt = source.transform()
    trans_id = tt.trans_id_tree_path('file1')
    tt.set_executability(True, trans_id)
    tt.apply()
    self.assertTrue(source.is_executable('file1'))
    target = self.make_branch_and_tree('target')
    revision_tree = source.basis_tree()
    revision_tree.lock_read()
    self.addCleanup(revision_tree.unlock)
    build_tree(revision_tree, target, source, hardlink=True)
    target.lock_read()
    self.addCleanup(target.unlock)
    self.assertEqual([], list(target.iter_changes(revision_tree)))
    self.assertTrue(source.is_executable('file1'))