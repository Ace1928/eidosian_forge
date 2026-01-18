from breezy.tests import TestNotApplicable
from breezy.tests.per_tree import TestCaseWithTree
from breezy.workingtree import SettingFileIdUnsupported
def test_get_root_id_fixed(self):
    try:
        tree = self.make_tree_with_fixed_root_id()
    except SettingFileIdUnsupported:
        raise TestNotApplicable('file ids not supported')
    with tree.lock_read():
        self.assertEqual(b'custom-tree-root-id', tree.path2id(''))