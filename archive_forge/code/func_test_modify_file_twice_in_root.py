import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_modify_file_twice_in_root(self):
    handler, branch = self.get_handler()
    path = b'a'
    handler.process(self.file_command_iter(path))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(path,)])
    self.assertContent(branch, revtree1, path, b'aaa')
    self.assertRevisionRoot(revtree1, path)