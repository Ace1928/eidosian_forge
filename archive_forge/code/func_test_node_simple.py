import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_node_simple(self):
    """
        Test a simple use of tgrep for finding nodes matching a given
        pattern.
        """
    tree = ParentedTree.fromstring('(S (NP (DT the) (JJ big) (NN dog)) (VP bit) (NP (DT a) (NN cat)))')
    self.assertEqual(list(tgrep.tgrep_positions('NN', [tree])), [[(0, 2), (2, 1)]])
    self.assertEqual(list(tgrep.tgrep_nodes('NN', [tree])), [[tree[0, 2], tree[2, 1]]])
    self.assertEqual(list(tgrep.tgrep_positions('NN|JJ', [tree])), [[(0, 1), (0, 2), (2, 1)]])