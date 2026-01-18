import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_delete_file_complex_level(self):
    handler, branch = self.get_handler()
    paths = [b'a/b/c', b'a/b/d/e', b'a/f/g', b'a/h', b'a/b/d/i/j']
    paths_to_delete = [b'a/b/c', b'a/b/d/e', b'a/f/g', b'a/b/d/i/j']
    handler.process(self.file_command_iter(paths, paths_to_delete))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (b'a/b/c',), (b'a/b/d',), (b'a/b/d/e',), (b'a/f',), (b'a/f/g',), (b'a/h',), (b'a/b/d/i',), (b'a/b/d/i/j',)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a/b',), (b'a/b/c',), (b'a/b/d',), (b'a/b/d/e',), (b'a/f',), (b'a/f/g',), (b'a/b/d/i',), (b'a/b/d/i/j',)])