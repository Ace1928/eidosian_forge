import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_TSP_incomplete_graph_short_path():
    G = nx.cycle_graph(9)
    G.add_edges_from([(4, 9), (9, 10), (10, 11), (11, 0)])
    G[4][5]['weight'] = 5
    cycle = nx_app.traveling_salesman_problem(G)
    print(cycle)
    assert len(cycle) == 17 and len(set(cycle)) == 12
    path = nx_app.traveling_salesman_problem(G, cycle=False)
    print(path)
    assert len(path) == 13 and len(set(path)) == 12