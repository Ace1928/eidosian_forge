import pytest
import networkx as nx
@staticmethod
def pick_add_edge(g):
    u = nx.utils.arbitrary_element(g)
    possible_nodes = set(g.nodes())
    neighbors = list(g.neighbors(u)) + [u]
    possible_nodes.difference_update(neighbors)
    v = nx.utils.arbitrary_element(possible_nodes)
    return (u, v)