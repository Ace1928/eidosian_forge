from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def dual_edges(overstrand, graph):
    """
    Find the set of crossings and edges of the dual graph encountered
    by moving along the link starting at startcep for length crossings.
    Also returns the next crossing entry point immediately after.
    """
    edges_crossed = []
    for cep in overstrand:
        f1 = graph.edge_to_face[cep]
        f2 = graph.edge_to_face[cep.opposite()]
        edges_crossed.append((f1, f2))
    endpoint = overstrand[-1].next()
    final_f1 = graph.edge_to_face[endpoint]
    final_f2 = graph.edge_to_face[endpoint.opposite()]
    edges_crossed.append((final_f1, final_f2))
    return edges_crossed