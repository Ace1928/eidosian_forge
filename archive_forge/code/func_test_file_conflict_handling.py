import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_file_conflict_handling(self):
    """Ensure that when building trees, conflict handling is done"""
    source = self.make_branch_and_tree('source')
    target = self.make_branch_and_tree('target')
    self.build_tree(['source/file', 'target/file'])
    source.add('file', ids=b'new-file')
    source.commit('added file')
    build_tree(source.basis_tree(), target)
    self.assertEqual([DuplicateEntry('Moved existing file to', 'file.moved', 'file', None, 'new-file')], target.conflicts())
    target2 = self.make_branch_and_tree('target2')
    with open('target2/file', 'wb') as target_file, open('source/file', 'rb') as source_file:
        target_file.write(source_file.read())
    build_tree(source.basis_tree(), target2)
    self.assertEqual([], target2.conflicts())