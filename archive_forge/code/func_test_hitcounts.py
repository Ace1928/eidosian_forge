import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
def test_hitcounts(self):
    rn = RenameMap(None)
    rn.add_edge_hashes(self.a_lines, 'a')
    rn.add_edge_hashes(self.b_lines, 'b')
    self.assertEqual({'a': 2.5, 'b': 0.5}, rn.hitcounts(self.a_lines))
    self.assertEqual({'a': 1}, rn.hitcounts(self.a_lines[:-1]))
    self.assertEqual({'b': 2.5, 'a': 0.5}, rn.hitcounts(self.b_lines))