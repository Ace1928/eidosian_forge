import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_build_tree_accelerator_tree(self):
    source = self.create_ab_tree()
    self.build_tree_contents([('source/file2', b'C')])
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
    self.assertEqual(['file1'], calls)
    target.lock_read()
    self.addCleanup(target.unlock)
    self.assertEqual([], list(target.iter_changes(revision_tree)))