import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_copy_file_in_subdir(self):
    handler, branch = self.get_handler()
    src_path = b'a/a'
    dest_path = b'a/b'
    handler.process(self.file_command_iter(src_path, dest_path))
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_added=[(dest_path,)])
    self.assertContent(branch, revtree1, src_path, b'aaa')
    self.assertContent(branch, revtree2, src_path, b'aaa')
    self.assertContent(branch, revtree2, dest_path, b'aaa')