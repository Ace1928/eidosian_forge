from itertools import chain
import networkx as nx
@nx._dispatch
def node_expansion(G, S):
    """Returns the node expansion of the set `S`.

    The *node expansion* is the quotient of the size of the node
    boundary of *S* and the cardinality of *S*. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    Returns
    -------
    number
        The node expansion of the set `S`.

    See also
    --------
    boundary_expansion
    edge_expansion
    mixing_expansion

    References
    ----------
    .. [1] Vadhan, Salil P.
           "Pseudorandomness."
           *Foundations and Trends
           in Theoretical Computer Science* 7.1–3 (2011): 1–336.
           <https://doi.org/10.1561/0400000010>

    """
    neighborhood = set(chain.from_iterable((G.neighbors(v) for v in S)))
    return len(neighborhood) / len(S)