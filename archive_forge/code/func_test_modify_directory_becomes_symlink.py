import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_modify_directory_becomes_symlink(self):
    handler, branch = self.get_handler()
    path1 = b'a/b/c'
    path2 = b'a/b'
    handler.process(self.file_command_iter(path1, path2, 'symlink'))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (path1,)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(path1,)], expected_kind_changed=[(path2, 'directory', 'symlink')])
    self.assertSymlinkTarget(branch, revtree1, path1, 'aaa')
    self.assertSymlinkTarget(branch, revtree2, path2, 'bbb')