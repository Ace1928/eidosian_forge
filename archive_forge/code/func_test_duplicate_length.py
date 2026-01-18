import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_duplicate_length(self):
    items = [10, 9, 10, 8, 9, 7, 8]
    s = sets.OrderedSet(iter(items))
    self.assertEqual(4, len(s))