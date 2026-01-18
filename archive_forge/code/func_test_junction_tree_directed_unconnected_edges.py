import networkx as nx
from networkx.algorithms.tree.decomposition import junction_tree
def test_junction_tree_directed_unconnected_edges():
    B = nx.DiGraph()
    B.add_edges_from([('A', 'B'), ('C', 'D'), ('E', 'F')])
    G = junction_tree(B)
    J = nx.Graph()
    J.add_nodes_from([('A', 'B'), ('C', 'D'), ('E', 'F')])
    assert nx.is_isomorphic(G, J)