import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_mixed_conflict_handling(self):
    """Ensure that when building trees, conflict handling is done"""
    source = self.make_branch_and_tree('source')
    target = self.make_branch_and_tree('target')
    self.build_tree(['source/name', 'target/name/'])
    source.add('name', ids=b'new-name')
    source.commit('added file')
    build_tree(source.basis_tree(), target)
    self.assertEqual([DuplicateEntry('Moved existing file to', 'name.moved', 'name', None, 'new-name')], target.conflicts())