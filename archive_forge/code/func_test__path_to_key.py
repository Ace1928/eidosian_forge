from breezy import multiwalker, revision
from breezy import tree as _mod_tree
from breezy.tests import TestCaseWithTransport
def test__path_to_key(self):
    self.assertPathToKey(([''], ''), '')
    self.assertPathToKey(([''], 'a'), 'a')
    self.assertPathToKey((['a'], 'b'), 'a/b')
    self.assertPathToKey((['a', 'b'], 'c'), 'a/b/c')