import networkx as nx
from networkx.algorithms.tree.decomposition import junction_tree
def test_junction_tree_undirected():
    B = nx.Graph()
    B.add_edges_from([('A', 'C'), ('A', 'D'), ('B', 'C'), ('C', 'E')])
    G = junction_tree(B)
    J = nx.Graph()
    J.add_edges_from([(('A', 'D'), ('A',)), (('A',), ('A', 'C')), (('A', 'C'), ('C',)), (('C',), ('B', 'C')), (('B', 'C'), ('C',)), (('C',), ('C', 'E'))])
    assert nx.is_isomorphic(G, J)