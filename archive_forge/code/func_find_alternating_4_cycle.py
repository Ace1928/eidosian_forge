from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
@nx._dispatch
def find_alternating_4_cycle(G):
    """
    Returns False if there aren't any alternating 4 cycles.
    Otherwise returns the cycle as [a,b,c,d] where (a,b)
    and (c,d) are edges and (a,c) and (b,d) are not.
    """
    for u, v in G.edges():
        for w in G.nodes():
            if not G.has_edge(u, w) and u != w:
                for x in G.neighbors(w):
                    if not G.has_edge(v, x) and v != x:
                        return [u, v, w, x]
    return False