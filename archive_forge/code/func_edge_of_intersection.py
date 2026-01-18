import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def edge_of_intersection(self, other):
    """
        Returns edge as the pair of CrossingStrands for self and other which
        defines the common edge. Returns None if the faces are disjoint.
        """
    common_edges = set(self.edges) & set(other.edges)
    if common_edges:
        e = common_edges.pop()
        return (self.edges[e], other.edges[e])