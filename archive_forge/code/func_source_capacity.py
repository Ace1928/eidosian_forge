import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def source_capacity(self):
    if self.exterior:
        return 0
    return max(4 - len(self), 0)