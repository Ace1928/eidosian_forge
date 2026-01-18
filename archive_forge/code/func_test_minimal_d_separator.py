from itertools import combinations
import pytest
import networkx as nx
def test_minimal_d_separator():
    edge_list = [('A', 'B'), ('C', 'B'), ('B', 'D'), ('D', 'E'), ('B', 'F'), ('G', 'E')]
    G = nx.DiGraph(edge_list)
    assert not nx.d_separated(G, {'B'}, {'E'}, set())
    Zmin = nx.minimal_d_separator(G, 'B', 'E')
    assert nx.is_minimal_d_separator(G, 'B', 'E', Zmin)
    assert Zmin == {'D'}
    edge_list = [('A', 'B'), ('B', 'C'), ('B', 'D'), ('D', 'C')]
    G = nx.DiGraph(edge_list)
    assert not nx.d_separated(G, {'A'}, {'C'}, set())
    Zmin = nx.minimal_d_separator(G, 'A', 'C')
    assert nx.is_minimal_d_separator(G, 'A', 'C', Zmin)
    assert Zmin == {'B'}
    Znotmin = Zmin.union({'D'})
    assert not nx.is_minimal_d_separator(G, 'A', 'C', Znotmin)