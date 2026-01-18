import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_empty_union(self):
    s = sets.OrderedSet([1, 2, 3])
    es = set(s)
    self.assertEqual(es.union(), s.union())