from itertools import chain
import networkx as nx
@nx._dispatch
def boundary_expansion(G, S):
    """Returns the boundary expansion of the set `S`.

    The *boundary expansion* is the quotient of the size
    of the node boundary and the cardinality of *S*. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    Returns
    -------
    number
        The boundary expansion of the set `S`.

    See also
    --------
    edge_expansion
    mixing_expansion
    node_expansion

    References
    ----------
    .. [1] Vadhan, Salil P.
           "Pseudorandomness."
           *Foundations and Trends in Theoretical Computer Science*
           7.1–3 (2011): 1–336.
           <https://doi.org/10.1561/0400000010>

    """
    return len(nx.node_boundary(G, S)) / len(S)