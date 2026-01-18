import pytest
import networkx as nx
def test_cycle_numpy():
    dist = nx.floyd_warshall_numpy(nx.cycle_graph(7))
    assert dist[0, 3] == 3
    assert dist[0, 4] == 3