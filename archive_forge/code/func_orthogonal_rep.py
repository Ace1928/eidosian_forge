import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def orthogonal_rep(self):
    orientations = self.orientations
    spec = [[(e.crossing, e.opposite().crossing) for e in self.edges if orientations[e] == dir] for dir in ['right', 'up']]
    return OrthogonalRep(*spec)