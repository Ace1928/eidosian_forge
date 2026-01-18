import pytest
import networkx as nx
def test_complement_2():
    G1 = nx.DiGraph()
    G1.add_edge('A', 'B')
    G1.add_edge('A', 'C')
    G1.add_edge('A', 'D')
    G1C = nx.complement(G1)
    assert sorted(G1C.edges()) == [('B', 'A'), ('B', 'C'), ('B', 'D'), ('C', 'A'), ('C', 'B'), ('C', 'D'), ('D', 'A'), ('D', 'B'), ('D', 'C')]