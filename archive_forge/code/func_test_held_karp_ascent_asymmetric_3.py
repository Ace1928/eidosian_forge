import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_held_karp_ascent_asymmetric_3():
    """
    Tests the ascent method using a truly asymmetric graph with a fractional
    solution for which the solution has been brute forced.

    In this graph their are two different optimal, integral solutions (which
    are also the overall atsp solutions) to the Held Karp relaxation. However,
    this particular graph has two different tours of optimal value and the
    possible solutions in the held_karp_ascent function are not stored in an
    ordered data structure.
    """
    import networkx.algorithms.approximation.traveling_salesman as tsp
    np = pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    G_array = np.array([[0, 1, 5, 2, 7, 4], [7, 0, 7, 7, 1, 4], [4, 7, 0, 9, 2, 1], [7, 2, 7, 0, 4, 4], [5, 5, 4, 4, 0, 3], [3, 9, 1, 3, 4, 0]])
    solution1_edges = [(0, 3), (1, 4), (2, 5), (3, 1), (4, 2), (5, 0)]
    solution2_edges = [(0, 3), (3, 1), (1, 4), (4, 5), (2, 0), (5, 2)]
    G = nx.from_numpy_array(G_array, create_using=nx.DiGraph)
    opt_hk, z_star = tsp.held_karp_ascent(G)
    assert round(opt_hk, 2) == 13.0
    solution1 = nx.DiGraph()
    solution1.add_edges_from(solution1_edges)
    solution2 = nx.DiGraph()
    solution2.add_edges_from(solution2_edges)
    assert nx.utils.edges_equal(z_star.edges, solution1.edges) or nx.utils.edges_equal(z_star.edges, solution2.edges)