import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_node_regex(self):
    """
        Test regex matching on nodes.
        """
    tree = ParentedTree.fromstring('(S (NP-SBJ x) (NP x) (NNP x) (VP x))')
    self.assertEqual(list(tgrep.tgrep_positions('/^NP/', [tree])), [[(0,), (1,)]])