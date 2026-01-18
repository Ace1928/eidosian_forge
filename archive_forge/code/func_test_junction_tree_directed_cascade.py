import networkx as nx
from networkx.algorithms.tree.decomposition import junction_tree
def test_junction_tree_directed_cascade():
    B = nx.DiGraph()
    B.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
    G = junction_tree(B)
    J = nx.Graph()
    J.add_edges_from([(('A', 'B'), ('B',)), (('B',), ('B', 'C')), (('B', 'C'), ('C',)), (('C',), ('C', 'D'))])
    assert nx.is_isomorphic(G, J)