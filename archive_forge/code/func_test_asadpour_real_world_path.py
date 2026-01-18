import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_asadpour_real_world_path():
    """
    This test uses airline prices between the six largest cities in the US. This
    time using a path, not a cycle.

        * New York City -> JFK
        * Los Angeles -> LAX
        * Chicago -> ORD
        * Houston -> IAH
        * Phoenix -> PHX
        * Philadelphia -> PHL

    Flight prices from August 2021 using Delta or American airlines to get
    nonstop flight. The brute force solution found the optimal tour to cost $872
    """
    np = pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    G_array = np.array([[0, 243, 199, 208, 169, 183], [277, 0, 217, 123, 127, 252], [297, 197, 0, 197, 123, 177], [303, 169, 197, 0, 117, 117], [257, 127, 160, 117, 0, 319], [183, 332, 217, 117, 319, 0]])
    node_map = {0: 'JFK', 1: 'LAX', 2: 'ORD', 3: 'IAH', 4: 'PHX', 5: 'PHL'}
    expected_paths = [['ORD', 'PHX', 'LAX', 'IAH', 'PHL', 'JFK'], ['JFK', 'PHL', 'IAH', 'ORD', 'PHX', 'LAX']]
    G = nx.from_numpy_array(G_array, create_using=nx.DiGraph)
    nx.relabel_nodes(G, node_map, copy=False)

    def fixed_asadpour(G, weight):
        return nx_app.asadpour_atsp(G, weight, 56)
    path = nx_app.traveling_salesman_problem(G, weight='weight', cycle=False, method=fixed_asadpour)
    assert path in expected_paths