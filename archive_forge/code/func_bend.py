import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def bend(self):
    """
        Computes a minimal size set of edge bends that allows the link diagram
        to be embedded orthogonally. This follows directly Tamassia's first
        paper.
        """
    N = self.face_network
    flow = networkx.min_cost_flow(N)
    for a, flows in flow.items():
        for b, w_a in flows.items():
            if w_a and set(['s', 't']).isdisjoint(set([a, b])):
                w_b = flow[b][a]
                A, B = (self[a], self[b])
                e_a, e_b = A.edge_of_intersection(B)
                turns_a = w_a * [1] + w_b * [-1]
                turns_b = w_b * [1] + w_a * [-1]
                subdivide_edge(e_a, len(turns_a))
                A.bend(e_a, turns_a)
                B.bend(e_b, turns_b)