import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def iterate_from(self, edge):
    i = self.index(edge)
    return zip(self[i:] + self[:i], self.turns[i:] + self.turns[:i])