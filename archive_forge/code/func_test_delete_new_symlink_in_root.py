import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_delete_new_symlink_in_root(self):
    handler, branch = self.get_handler()
    path = b'a'
    handler.process(self.file_command_iter(path, kind='symlink'))
    revtree0, revtree1 = self.assertChanges(branch, 1)