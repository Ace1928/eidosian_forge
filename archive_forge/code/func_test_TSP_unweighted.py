import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_TSP_unweighted():
    G = nx.cycle_graph(9)
    path = nx_app.traveling_salesman_problem(G, nodes=[3, 6], cycle=False)
    assert path in ([3, 4, 5, 6], [6, 5, 4, 3])
    cycle = nx_app.traveling_salesman_problem(G, nodes=[3, 6])
    assert cycle in ([3, 4, 5, 6, 5, 4, 3], [6, 5, 4, 3, 4, 5, 6])