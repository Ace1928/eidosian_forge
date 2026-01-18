import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def orthogonal_spec(self):
    orientations = self.orientations
    return [[(e.crossing.label, e.opposite().crossing.label) for e in self.edges if orientations[e] == dir] for dir in ['right', 'up']]