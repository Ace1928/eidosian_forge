import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_node_nocase(self):
    """
        Test selecting nodes using case insensitive node names.
        """
    tree = ParentedTree.fromstring('(S (n x) (N x))')
    self.assertEqual(list(tgrep.tgrep_positions('"N"', [tree])), [[(1,)]])
    self.assertEqual(list(tgrep.tgrep_positions('i@"N"', [tree])), [[(0,), (1,)]])