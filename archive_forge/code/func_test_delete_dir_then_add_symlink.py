import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_delete_dir_then_add_symlink(self):
    handler, branch = self.get_handler()
    paths = [b'a/b/c', b'a/b/d']
    dir = b'a/b'
    new_path = b'a/b/z'
    handler.process(self.file_command_iter(paths, dir, new_path, 'symlink'))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (b'a/b/c',), (b'a/b/d',)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a/b',), (b'a/b/c',), (b'a/b/d',)], expected_added=[(b'a/b',), (b'a/b/z',)])
    self.assertSymlinkTarget(branch, revtree2, new_path, 'bbb')