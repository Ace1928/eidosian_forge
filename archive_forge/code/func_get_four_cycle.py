import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def get_four_cycle(G, start_vertex):
    """
    Returns the first nontrivial 4-cycle found in the graph G
    """
    adjacent = G.children(start_vertex)
    for v in adjacent:
        for w in adjacent:
            if v == w:
                continue
            new_adj = G.children(v).intersection(G.children(w))
            new_adj.remove(start_vertex)
            for opposite_vertex in new_adj:
                for e1 in G.edges_between(start_vertex, v):
                    for e2 in G.edges_between(v, opposite_vertex):
                        for e3 in G.edges_between(opposite_vertex, w):
                            for e4 in G.edges_between(w, start_vertex):
                                four_cycle = (e1, e2, e3, e4)
                                if is_trivial(four_cycle):
                                    continue
                                else:
                                    return four_cycle
    return []