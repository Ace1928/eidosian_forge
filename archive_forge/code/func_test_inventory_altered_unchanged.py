import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_inventory_altered_unchanged(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/foo'])
    tree.add('foo', ids=b'foo-id')
    with tree.preview_transform() as tt:
        self.assertEqual([], tt._inventory_altered())