import pytest
import networkx as nx
from networkx.generators import directed
def test_iteration_depth_edge_attr():
    """
    All nodes should have the correct number of subgraph hashes in the output when
    setting initial node labels empty and using an edge attribute when aggregating
    neighborhoods.
    Subsequent iteration depths for the same graph should be additive for each node
    """
    n, r = (100, 10)
    p = 1.0 / r
    for i in range(1, r + 1):
        G = nx.erdos_renyi_graph(n, p * i, seed=700 + i)
        for a, b in G.edges:
            G[a][b]['edge_attr1'] = f'{a}-{b}-1'
        depth3 = nx.weisfeiler_lehman_subgraph_hashes(G, edge_attr='edge_attr1', iterations=3)
        depth4 = nx.weisfeiler_lehman_subgraph_hashes(G, edge_attr='edge_attr1', iterations=4)
        depth5 = nx.weisfeiler_lehman_subgraph_hashes(G, edge_attr='edge_attr1', iterations=5)
        assert all((len(hashes) == 3 for hashes in depth3.values()))
        assert all((len(hashes) == 4 for hashes in depth4.values()))
        assert all((len(hashes) == 5 for hashes in depth5.values()))
        assert is_subiteration(depth3, depth4)
        assert is_subiteration(depth4, depth5)
        assert is_subiteration(depth3, depth5)