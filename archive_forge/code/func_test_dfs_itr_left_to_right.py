import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_dfs_itr_left_to_right(self):
    root = self._make_species()
    it = root.dfs_iter(include_self=False, right_to_left=False)
    things = list([n.item for n in it])
    self.assertEqual(['reptile', 'mammal', 'primate', 'human', 'monkey', 'horse'], things)