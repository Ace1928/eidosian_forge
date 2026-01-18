import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_node_printing(self):
    """Test that the tgrep print operator ' is properly ignored."""
    tree = ParentedTree.fromstring('(S (n x) (N x))')
    self.assertEqual(list(tgrep.tgrep_positions('N', [tree])), list(tgrep.tgrep_positions("'N", [tree])))
    self.assertEqual(list(tgrep.tgrep_positions('/[Nn]/', [tree])), list(tgrep.tgrep_positions("'/[Nn]/", [tree])))