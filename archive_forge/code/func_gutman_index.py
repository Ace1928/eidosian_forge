import itertools as it
import networkx as nx
@nx.utils.not_implemented_for('directed')
@nx.utils.not_implemented_for('multigraph')
@nx._dispatchable(edge_attrs='weight')
def gutman_index(G, weight=None):
    """Returns the Gutman Index for the graph `G`.

    The *Gutman Index* measures the topology of networks, especially for molecule
    networks of atoms connected by bonds [1]_. It is also called the Schultz Index
    of the second kind [2]_.

    Consider an undirected graph `G` with node set ``V``.
    The Gutman Index of a graph is the sum over all (unordered) pairs of nodes
    of nodes ``(u, v)``, with distance ``dist(u, v)`` and degrees ``deg(u)``
    and ``deg(v)``, of ``dist(u, v) * deg(u) * deg(v)``

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default: None)
        If None, every edge has weight 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        The edge weights are used to computing shortest-path distances.

    Returns
    -------
    number
        The Gutman Index of the graph `G`.

    Examples
    --------
    The Gutman Index of the (unweighted) complete graph on *n* nodes
    equals the number of pairs of the *n* nodes times ``(n - 1) * (n - 1)``,
    since each pair of nodes is at distance one and the product of degree of two
    vertices is ``(n - 1) * (n - 1)``.

    >>> n = 10
    >>> G = nx.complete_graph(n)
    >>> nx.gutman_index(G) == (n * (n - 1) / 2) * ((n - 1) * (n - 1))
    True

    Graphs that are disconnected

    >>> G = nx.empty_graph(2)
    >>> nx.gutman_index(G)
    inf

    References
    ----------
    .. [1] M.V. Diudeaa and I. Gutman, Wiener-Type Topological Indices,
           Croatica Chemica Acta, 71 (1998), 21-51.
           https://hrcak.srce.hr/132323
    .. [2] I. Gutman, Selected properties of the Schultz molecular topological index,
           J. Chem. Inf. Comput. Sci. 34 (1994), 1087–1089.
           https://doi.org/10.1021/ci00021a009

    """
    if not nx.is_connected(G):
        return float('inf')
    spl = nx.shortest_path_length(G, weight=weight)
    d = dict(G.degree, weight=weight)
    return sum((dist * d[u] * d[v] for u, vinfo in spl for v, dist in vinfo.items())) / 2