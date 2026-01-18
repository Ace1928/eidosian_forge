import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def sink_capacity(self):
    if self.exterior:
        return len(self) + 4
    return max(len(self) - 4, 0)