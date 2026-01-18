import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_after_frozen(self):
    root = tree.Node('josh')
    root.add(tree.Node('josh.1'))
    root.freeze()
    self.assertTrue(all((n.frozen for n in root.dfs_iter(include_self=True))))
    self.assertRaises(tree.FrozenNode, root.remove, 'josh.1')
    self.assertRaises(tree.FrozenNode, root.disassociate)
    self.assertRaises(tree.FrozenNode, root.add, tree.Node('josh.2'))