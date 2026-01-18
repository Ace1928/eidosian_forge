import networkx as nx
from networkx.algorithms.centrality.betweenness import (
from networkx.algorithms.centrality.betweenness import (
from networkx.algorithms.centrality.betweenness import (
@nx._dispatch(edge_attrs='weight')
def betweenness_centrality_subset(G, sources, targets, normalized=False, weight=None):
    """Compute betweenness centrality for a subset of nodes.

    .. math::

       c_B(v) =\\sum_{s\\in S, t \\in T} \\frac{\\sigma(s, t|v)}{\\sigma(s, t)}

    where $S$ is the set of sources, $T$ is the set of targets,
    $\\sigma(s, t)$ is the number of shortest $(s, t)$-paths,
    and $\\sigma(s, t|v)$ is the number of those paths
    passing through some  node $v$ other than $s, t$.
    If $s = t$, $\\sigma(s, t) = 1$,
    and if $v \\in {s, t}$, $\\sigma(s, t|v) = 0$ [2]_.


    Parameters
    ----------
    G : graph
      A NetworkX graph.

    sources: list of nodes
      Nodes to use as sources for shortest paths in betweenness

    targets: list of nodes
      Nodes to use as targets for shortest paths in betweenness

    normalized : bool, optional
      If True the betweenness values are normalized by $2/((n-1)(n-2))$
      for graphs, and $1/((n-1)(n-2))$ for directed graphs where $n$
      is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      Weights are used to calculate weighted shortest paths, so they are
      interpreted as distances.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value.

    See Also
    --------
    edge_betweenness_centrality
    load_centrality

    Notes
    -----
    The basic algorithm is from [1]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    The normalization might seem a little strange but it is
    designed to make betweenness_centrality(G) be the same as
    betweenness_centrality_subset(G,sources=G.nodes(),targets=G.nodes()).

    The total number of paths between source and target is counted
    differently for directed and undirected graphs. Directed paths
    are easy to count. Undirected paths are tricky: should a path
    from "u" to "v" count as 1 undirected path or as 2 directed paths?

    For betweenness_centrality we report the number of undirected
    paths when G is undirected.

    For betweenness_centrality_subset the reporting is different.
    If the source and target subsets are the same, then we want
    to count undirected paths. But if the source and target subsets
    differ -- for example, if sources is {0} and targets is {1},
    then we are only counting the paths in one direction. They are
    undirected paths but we are counting them in a directed way.
    To count them as undirected paths, each should count as half a path.

    References
    ----------
    .. [1] Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001
    """
    b = dict.fromkeys(G, 0.0)
    for s in sources:
        if weight is None:
            S, P, sigma, _ = shortest_path(G, s)
        else:
            S, P, sigma, _ = dijkstra(G, s, weight)
        b = _accumulate_subset(b, S, P, sigma, s, targets)
    b = _rescale(b, len(G), normalized=normalized, directed=G.is_directed())
    return b