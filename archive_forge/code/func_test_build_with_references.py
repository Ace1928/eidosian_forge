import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_build_with_references(self):
    tree = self.make_branch_and_tree('source', format='development-subtree')
    subtree = self.make_branch_and_tree('source/subtree', format='development-subtree')
    tree.add_reference(subtree)
    tree.commit('a revision')
    tree.branch.create_checkout('target')
    self.assertPathExists('target')
    self.assertPathExists('target/subtree')