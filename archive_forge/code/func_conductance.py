from itertools import chain
import networkx as nx
@nx._dispatch(edge_attrs='weight')
def conductance(G, S, T=None, weight=None):
    """Returns the conductance of two sets of nodes.

    The *conductance* is the quotient of the cut size and the smaller of
    the volumes of the two sets. [1]

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
        The conductance between the two sets `S` and `T`.

    See also
    --------
    cut_size
    edge_expansion
    normalized_cut_size
    volume

    References
    ----------
    .. [1] David Gleich.
           *Hierarchical Directed Spectral Graph Partitioning*.
           <https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20hierarchical%20directed%20spectral.pdf>

    """
    if T is None:
        T = set(G) - set(S)
    num_cut_edges = cut_size(G, S, T, weight=weight)
    volume_S = volume(G, S, weight=weight)
    volume_T = volume(G, T, weight=weight)
    return num_cut_edges / min(volume_S, volume_T)