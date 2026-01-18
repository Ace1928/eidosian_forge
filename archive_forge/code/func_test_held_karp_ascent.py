import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_held_karp_ascent():
    """
    Test the Held-Karp relaxation with the ascent method
    """
    import networkx.algorithms.approximation.traveling_salesman as tsp
    np = pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    G_array = np.array([[0, 97, 60, 73, 17, 52], [97, 0, 41, 52, 90, 30], [60, 41, 0, 21, 35, 41], [73, 52, 21, 0, 95, 46], [17, 90, 35, 95, 0, 81], [52, 30, 41, 46, 81, 0]])
    solution_edges = [(1, 3), (2, 4), (3, 2), (4, 0), (5, 1), (0, 5)]
    G = nx.from_numpy_array(G_array, create_using=nx.DiGraph)
    opt_hk, z_star = tsp.held_karp_ascent(G)
    assert round(opt_hk, 2) == 207.0
    solution = nx.DiGraph()
    solution.add_edges_from(solution_edges)
    assert nx.utils.edges_equal(z_star.edges, solution.edges)