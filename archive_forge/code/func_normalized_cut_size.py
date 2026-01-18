from itertools import chain
import networkx as nx
@nx._dispatch(edge_attrs='weight')
def normalized_cut_size(G, S, T=None, weight=None):
    """Returns the normalized size of the cut between two sets of nodes.

    The *normalized cut size* is the cut size times the sum of the
    reciprocal sizes of the volumes of the two sets. [1]

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
        The normalized cut size between the two sets `S` and `T`.

    Notes
    -----
    In a multigraph, the cut size is the total weight of edges including
    multiplicity.

    See also
    --------
    conductance
    cut_size
    edge_expansion
    volume

    References
    ----------
    .. [1] David Gleich.
           *Hierarchical Directed Spectral Graph Partitioning*.
           <https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20hierarchical%20directed%20spectral.pdf>

    """
    if T is None:
        T = set(G) - set(S)
    num_cut_edges = cut_size(G, S, T=T, weight=weight)
    volume_S = volume(G, S, weight=weight)
    volume_T = volume(G, T, weight=weight)
    return num_cut_edges * (1 / volume_S + 1 / volume_T)