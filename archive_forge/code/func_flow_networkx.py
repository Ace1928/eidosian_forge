import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def flow_networkx(faces):
    """
        Tamassia's associated graph N(P) where the flow problem resides.
        """
    G = networkx.DiGraph()
    source_demand = sum((F.source_capacity() for F in faces))
    G.add_node('s', demand=-source_demand)
    for i, F in enumerate(faces):
        if F.source_capacity():
            G.add_edge('s', i, weight=0, capacity=F.source_capacity())
    sink_demand = sum((F.sink_capacity() for F in faces))
    assert sink_demand == source_demand
    G.add_node('t', demand=sink_demand)
    for i, F in enumerate(faces):
        if F.sink_capacity():
            G.add_edge(i, 't', weight=0, capacity=F.sink_capacity())
    for A in faces:
        for B in faces:
            if A != B and A.edge_of_intersection(B):
                G.add_edge(faces.index(A), faces.index(B), weight=1)
    return G