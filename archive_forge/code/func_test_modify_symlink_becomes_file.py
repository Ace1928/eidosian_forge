import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_modify_symlink_becomes_file(self):
    handler, branch = self.get_handler()
    path = b'a/a'
    handler.process(self.file_command_iter(path, kind='symlink', to_kind='file'))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_kind_changed=[(path, 'symlink', 'file')])
    self.assertSymlinkTarget(branch, revtree1, path, 'aaa')
    self.assertContent(branch, revtree2, path, b'bbb')