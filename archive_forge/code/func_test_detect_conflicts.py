import os
from ... import conflicts, errors
from ...bzr.conflicts import ContentsConflict, TextConflict
from ...tests import TestCaseWithTransport
from .bzrdir import BzrDirFormat6
def test_detect_conflicts(self):
    """Conflicts are detected properly"""
    tree = self.create_format2_tree('.')
    self.build_tree_contents([('hello', b'hello world4'), ('hello.THIS', b'hello world2'), ('hello.BASE', b'hello world1'), ('hello.OTHER', b'hello world3'), ('hello.sploo.BASE', b'yellowworld'), ('hello.sploo.OTHER', b'yellowworld2')])
    tree.lock_read()
    self.assertLength(6, list(tree.list_files()))
    tree.unlock()
    tree_conflicts = tree.conflicts()
    self.assertLength(2, tree_conflicts)
    self.assertTrue('hello' in tree_conflicts[0].path)
    self.assertTrue('hello.sploo' in tree_conflicts[1].path)
    conflicts.restore('hello')
    conflicts.restore('hello.sploo')
    self.assertLength(0, tree.conflicts())
    self.assertFileEqual(b'hello world2', 'hello')
    self.assertFalse(os.path.lexists('hello.sploo'))
    self.assertRaises(errors.NotConflicted, conflicts.restore, 'hello')
    self.assertRaises(errors.NotConflicted, conflicts.restore, 'hello.sploo')