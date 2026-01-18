import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def saturation_edges(self, swap_hor_edges):
    return sum([face.saturation_edges(swap_hor_edges) for face in self.faces], [])