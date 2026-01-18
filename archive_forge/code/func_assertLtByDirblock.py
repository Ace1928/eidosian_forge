from breezy import multiwalker, revision
from breezy import tree as _mod_tree
from breezy.tests import TestCaseWithTransport
def assertLtByDirblock(self, lt_val, path1, path2):
    self.assertEqual(lt_val, multiwalker.MultiWalker._lt_path_by_dirblock(path1, path2))