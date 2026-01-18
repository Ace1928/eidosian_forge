import networkx as nx
from networkx.algorithms.tree.decomposition import junction_tree
def test_junction_tree_directed_confounders():
    B = nx.DiGraph()
    B.add_edges_from([('A', 'C'), ('B', 'C'), ('C', 'D'), ('C', 'E')])
    G = junction_tree(B)
    J = nx.Graph()
    J.add_edges_from([(('C', 'E'), ('C',)), (('C',), ('A', 'B', 'C')), (('A', 'B', 'C'), ('C',)), (('C',), ('C', 'D'))])
    assert nx.is_isomorphic(G, J)