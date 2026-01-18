import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_inventory_altered_noop_changed_parent_id(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/foo'])
    tree.add('foo', ids=b'foo-id')
    with tree.preview_transform() as tt:
        tt.unversion_file(tt.root)
        tt.version_file(tt.root, file_id=tree.path2id(''))
        tt.trans_id_tree_path('foo')
        self.assertEqual([], tt._inventory_altered())