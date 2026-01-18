import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_bfs_itr_right_to_left(self):
    root = self._make_species()
    it = root.bfs_iter(include_self=False, right_to_left=True)
    things = list([n.item for n in it])
    self.assertEqual(['mammal', 'reptile', 'horse', 'primate', 'monkey', 'human'], things)