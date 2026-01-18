from itertools import combinations
import networkx as nx
from networkx.algorithms.simple_paths import is_simple_path as is_path
from networkx.utils import arbitrary_element, not_implemented_for, py_random_state
@not_implemented_for('undirected')
@not_implemented_for('multigraph')
@nx._dispatch
def score_sequence(G):
    """Returns the score sequence for the given tournament graph.

    The score sequence is the sorted list of the out-degrees of the
    nodes of the graph.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph representing a tournament.

    Returns
    -------
    list
        A sorted list of the out-degrees of the nodes of `G`.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 0), (1, 3), (0, 2), (0, 3), (2, 1), (3, 2)])
    >>> nx.is_tournament(G)
    True
    >>> nx.tournament.score_sequence(G)
    [1, 1, 2, 2]

    """
    return sorted((d for v, d in G.out_degree()))