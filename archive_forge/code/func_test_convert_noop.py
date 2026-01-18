import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_convert_noop(self):
    t = timing.convert_to_timeout(1.0)
    t2 = timing.convert_to_timeout(t)
    self.assertEqual(t, t2)