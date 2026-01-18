from breezy import multiwalker, revision
from breezy import tree as _mod_tree
from breezy.tests import TestCaseWithTransport
def test__lt_path_by_dirblock(self):
    self.assertRaises(TypeError, multiwalker.MultiWalker._lt_path_by_dirblock, b'', b'b')
    self.assertLtByDirblock(False, '', '')
    self.assertLtByDirblock(False, 'a', 'a')
    self.assertLtByDirblock(False, 'a/b', 'a/b')
    self.assertLtByDirblock(False, 'a/b/c', 'a/b/c')
    self.assertLtByDirblock(False, 'a-a', 'a')
    self.assertLtByDirblock(True, 'a-a', 'a/a')
    self.assertLtByDirblock(True, 'a=a', 'a/a')
    self.assertLtByDirblock(False, 'a-a/a', 'a/a')
    self.assertLtByDirblock(False, 'a=a/a', 'a/a')
    self.assertLtByDirblock(False, 'a-a/a', 'a/a/a')
    self.assertLtByDirblock(False, 'a=a/a', 'a/a/a')
    self.assertLtByDirblock(False, 'a-a/a/a', 'a/a/a')
    self.assertLtByDirblock(False, 'a=a/a/a', 'a/a/a')