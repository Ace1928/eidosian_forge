import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_modify_file_no_longer_executable(self):
    handler, branch = self.get_handler()
    path = b'a/a'
    handler.process(self.file_command_iter(path, executable=True, to_executable=False, to_content=b'aaa'))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(path,)])
    self.assertExecutable(branch, revtree1, path, True)
    self.assertExecutable(branch, revtree2, path, False)