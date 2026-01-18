import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def edge_cycle(vert_list, G):
    """
    Converts from list of vertices of dual graph to list of edges.
    If multiple edges, just chooses one.
    """
    edges = list(G.edges)
    cycle = []
    for i in range(len(vert_list) - 1):
        face_pair = [vert_list[i], vert_list[i + 1]]
        for edge in edges:
            if set(face_pair) == set(edge.incident_to()):
                cycle.append(edge)
                break
    face_pair = [vert_list[0], vert_list[-1]]
    for edge in edges:
        if set(face_pair) == set(edge.incident_to()):
            cycle.append(edge)
            break
    return cycle