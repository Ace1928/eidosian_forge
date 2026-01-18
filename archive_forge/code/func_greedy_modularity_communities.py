from collections import defaultdict
import networkx as nx
from networkx.algorithms.community.quality import modularity
from networkx.utils import not_implemented_for
from networkx.utils.mapped_queue import MappedQueue
@nx._dispatch(edge_attrs='weight')
def greedy_modularity_communities(G, weight=None, resolution=1, cutoff=1, best_n=None):
    """Find communities in G using greedy modularity maximization.

    This function uses Clauset-Newman-Moore greedy modularity maximization [2]_
    to find the community partition with the largest modularity.

    Greedy modularity maximization begins with each node in its own community
    and repeatedly joins the pair of communities that lead to the largest
    modularity until no further increase in modularity is possible (a maximum).
    Two keyword arguments adjust the stopping condition. `cutoff` is a lower
    limit on the number of communities so you can stop the process before
    reaching a maximum (used to save computation time). `best_n` is an upper
    limit on the number of communities so you can make the process continue
    until at most n communities remain even if the maximum modularity occurs
    for more. To obtain exactly n communities, set both `cutoff` and `best_n` to n.

    This function maximizes the generalized modularity, where `resolution`
    is the resolution parameter, often expressed as $\\gamma$.
    See :func:`~networkx.algorithms.community.quality.modularity`.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default=None)
        The name of an edge attribute that holds the numerical value used
        as a weight.  If None, then each edge has weight 1.
        The degree is the sum of the edge weights adjacent to the node.

    resolution : float, optional (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    cutoff : int, optional (default=1)
        A minimum number of communities below which the merging process stops.
        The process stops at this number of communities even if modularity
        is not maximized. The goal is to let the user stop the process early.
        The process stops before the cutoff if it finds a maximum of modularity.

    best_n : int or None, optional (default=None)
        A maximum number of communities above which the merging process will
        not stop. This forces community merging to continue after modularity
        starts to decrease until `best_n` communities remain.
        If ``None``, don't force it to continue beyond a maximum.

    Raises
    ------
    ValueError : If the `cutoff` or `best_n`  value is not in the range
        ``[1, G.number_of_nodes()]``, or if `best_n` < `cutoff`.

    Returns
    -------
    communities: list
        A list of frozensets of nodes, one for each community.
        Sorted by length with largest communities first.

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> c = nx.community.greedy_modularity_communities(G)
    >>> sorted(c[0])
    [8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    See Also
    --------
    modularity

    References
    ----------
    .. [1] Newman, M. E. J. "Networks: An Introduction", page 224
       Oxford University Press 2011.
    .. [2] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.
    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community
       Detection" Phys. Rev. E74, 2006.
    .. [4] Newman, M. E. J."Analysis of weighted networks"
       Physical Review E 70(5 Pt 2):056131, 2004.
    """
    if cutoff < 1 or cutoff > G.number_of_nodes():
        raise ValueError(f'cutoff must be between 1 and {len(G)}. Got {cutoff}.')
    if best_n is not None:
        if best_n < 1 or best_n > G.number_of_nodes():
            raise ValueError(f'best_n must be between 1 and {len(G)}. Got {best_n}.')
        if best_n < cutoff:
            raise ValueError(f'Must have best_n >= cutoff. Got {best_n} < {cutoff}')
        if best_n == 1:
            return [set(G)]
    else:
        best_n = G.number_of_nodes()
    community_gen = _greedy_modularity_communities_generator(G, weight=weight, resolution=resolution)
    communities = next(community_gen)
    while len(communities) > cutoff:
        try:
            dq = next(community_gen)
        except StopIteration:
            communities = sorted(communities, key=len, reverse=True)
            while len(communities) > best_n:
                comm1, comm2, *rest = communities
                communities = [comm1 ^ comm2]
                communities.extend(rest)
            return communities
        if dq < 0 and len(communities) <= best_n:
            break
        communities = next(community_gen)
    return sorted(communities, key=len, reverse=True)