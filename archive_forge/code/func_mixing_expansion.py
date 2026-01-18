from itertools import chain
import networkx as nx
@nx._dispatch(edge_attrs='weight')
def mixing_expansion(G, S, T=None, weight=None):
    """Returns the mixing expansion between two node sets.

    The *mixing expansion* is the quotient of the cut size and twice the
    number of edges in the graph. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    T : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The mixing expansion between the two sets `S` and `T`.

    See also
    --------
    boundary_expansion
    edge_expansion
    node_expansion

    References
    ----------
    .. [1] Vadhan, Salil P.
           "Pseudorandomness."
           *Foundations and Trends
           in Theoretical Computer Science* 7.1–3 (2011): 1–336.
           <https://doi.org/10.1561/0400000010>

    """
    num_cut_edges = cut_size(G, S, T=T, weight=weight)
    num_total_edges = G.number_of_edges()
    return num_cut_edges / (2 * num_total_edges)