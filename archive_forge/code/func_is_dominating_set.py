from itertools import chain
import networkx as nx
from networkx.utils import arbitrary_element
@nx._dispatch
def is_dominating_set(G, nbunch):
    """Checks if `nbunch` is a dominating set for `G`.

    A *dominating set* for a graph with node set *V* is a subset *D* of
    *V* such that every node not in *D* is adjacent to at least one
    member of *D* [1]_.

    Parameters
    ----------
    G : NetworkX graph

    nbunch : iterable
        An iterable of nodes in the graph `G`.

    See also
    --------
    dominating_set

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dominating_set

    """
    testset = {n for n in nbunch if n in G}
    nbrs = set(chain.from_iterable((G[n] for n in testset)))
    return len(set(G) - testset - nbrs) == 0