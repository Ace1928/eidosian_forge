import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_copy_new_symlink_in_root(self):
    handler, branch = self.get_handler()
    src_path = b'a'
    dest_path = b'b'
    handler.process(self.file_command_iter(src_path, dest_path, 'symlink'))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(src_path,), (dest_path,)])
    self.assertSymlinkTarget(branch, revtree1, src_path, 'aaa')
    self.assertSymlinkTarget(branch, revtree1, dest_path, 'aaa')
    self.assertRevisionRoot(revtree1, src_path)
    self.assertRevisionRoot(revtree1, dest_path)