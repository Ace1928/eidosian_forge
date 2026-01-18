import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
@nx._dispatchable(graphs=None, returns_graph=True)
def kneser_graph(n, k):
    """Returns the Kneser Graph with parameters `n` and `k`.

    The Kneser Graph has nodes that are k-tuples (subsets) of the integers
    between 0 and ``n-1``. Nodes are adjacent if their corresponding sets are disjoint.

    Parameters
    ----------
    n: int
        Number of integers from which to make node subsets.
        Subsets are drawn from ``set(range(n))``.
    k: int
        Size of the subsets.

    Returns
    -------
    G : NetworkX Graph

    Examples
    --------
    >>> G = nx.kneser_graph(5, 2)
    >>> G.number_of_nodes()
    10
    >>> G.number_of_edges()
    15
    >>> nx.is_isomorphic(G, nx.petersen_graph())
    True
    """
    if n <= 0:
        raise NetworkXError('n should be greater than zero')
    if k <= 0 or k > n:
        raise NetworkXError('k should be greater than zero and smaller than n')
    G = nx.Graph()
    subsets = list(itertools.combinations(range(n), k))
    if 2 * k > n:
        G.add_nodes_from(subsets)
    universe = set(range(n))
    comb = itertools.combinations
    G.add_edges_from(((s, t) for s in subsets for t in comb(universe - set(s), k)))
    return G