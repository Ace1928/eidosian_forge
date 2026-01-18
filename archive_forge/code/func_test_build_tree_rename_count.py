import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_build_tree_rename_count(self):
    source = self.make_branch_and_tree('source')
    self.build_tree(['source/file1', 'source/dir1/'])
    source.add(['file1', 'dir1'])
    source.commit('add1')
    target1 = self.make_branch_and_tree('target1')
    transform_result = build_tree(source.basis_tree(), target1)
    self.assertEqual(2, transform_result.rename_count)
    self.build_tree(['source/dir1/file2'])
    source.add(['dir1/file2'])
    source.commit('add3')
    target2 = self.make_branch_and_tree('target2')
    transform_result = build_tree(source.basis_tree(), target2)
    self.assertEqual(2, transform_result.rename_count)