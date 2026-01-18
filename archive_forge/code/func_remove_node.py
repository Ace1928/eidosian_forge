import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
def remove_node(self, n):
    keys = set()
    for keydict in self.pred[n].values():
        keys.update(keydict)
    for keydict in self.succ[n].values():
        keys.update(keydict)
    for key in keys:
        del self.edge_index[key]
    self._cls.remove_node(n)