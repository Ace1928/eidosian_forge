import itertools
import networkx as nx
from networkx.algorithms.flow import build_residual_network, edmonds_karp
from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity
@nx._dispatch(graphs={'G': 0, 'auxiliary?': 4, 'residual?': 5}, preserve_edge_attrs={'auxiliary': {'capacity': float('inf')}, 'residual': {'capacity': float('inf')}}, preserve_graph_attrs={'auxiliary', 'residual'})
def minimum_st_edge_cut(G, s, t, flow_func=None, auxiliary=None, residual=None):
    """Returns the edges of the cut-set of a minimum (s, t)-cut.

    This function returns the set of edges of minimum cardinality that,
    if removed, would destroy all paths among source and target in G.
    Edge weights are not considered. See :meth:`minimum_cut` for
    computing minimum cuts considering edge weights.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node for the flow.

    t : node
        Sink node for the flow.

    auxiliary : NetworkX DiGraph
        Auxiliary digraph to compute flow based node connectivity. It has
        to have a graph attribute called mapping with a dictionary mapping
        node names in G and in the auxiliary digraph. If provided
        it will be reused instead of recreated. Default value: None.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See :meth:`node_connectivity` for
        details. The choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    residual : NetworkX DiGraph
        Residual network to compute maximum flow. If provided it will be
        reused instead of recreated. Default value: None.

    Returns
    -------
    cutset : set
        Set of edges that, if removed from the graph, will disconnect it.

    See also
    --------
    :meth:`minimum_cut`
    :meth:`minimum_node_cut`
    :meth:`minimum_edge_cut`
    :meth:`stoer_wagner`
    :meth:`node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Examples
    --------
    This function is not imported in the base NetworkX namespace, so you
    have to explicitly import it from the connectivity package:

    >>> from networkx.algorithms.connectivity import minimum_st_edge_cut

    We use in this example the platonic icosahedral graph, which has edge
    connectivity 5.

    >>> G = nx.icosahedral_graph()
    >>> len(minimum_st_edge_cut(G, 0, 6))
    5

    If you need to compute local edge cuts on several pairs of
    nodes in the same graph, it is recommended that you reuse the
    data structures that NetworkX uses in the computation: the
    auxiliary digraph for edge connectivity, and the residual
    network for the underlying maximum flow computation.

    Example of how to compute local edge cuts among all pairs of
    nodes of the platonic icosahedral graph reusing the data
    structures.

    >>> import itertools
    >>> # You also have to explicitly import the function for
    >>> # building the auxiliary digraph from the connectivity package
    >>> from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity
    >>> H = build_auxiliary_edge_connectivity(G)
    >>> # And the function for building the residual network from the
    >>> # flow package
    >>> from networkx.algorithms.flow import build_residual_network
    >>> # Note that the auxiliary digraph has an edge attribute named capacity
    >>> R = build_residual_network(H, "capacity")
    >>> result = dict.fromkeys(G, dict())
    >>> # Reuse the auxiliary digraph and the residual network by passing them
    >>> # as parameters
    >>> for u, v in itertools.combinations(G, 2):
    ...     k = len(minimum_st_edge_cut(G, u, v, auxiliary=H, residual=R))
    ...     result[u][v] = k
    >>> all(result[u][v] == 5 for u, v in itertools.combinations(G, 2))
    True

    You can also use alternative flow algorithms for computing edge
    cuts. For instance, in dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better than
    the default :meth:`edmonds_karp` which is faster for sparse
    networks with highly skewed degree distributions. Alternative flow
    functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> len(minimum_st_edge_cut(G, 0, 6, flow_func=shortest_augmenting_path))
    5

    """
    if flow_func is None:
        flow_func = default_flow_func
    if auxiliary is None:
        H = build_auxiliary_edge_connectivity(G)
    else:
        H = auxiliary
    kwargs = {'capacity': 'capacity', 'flow_func': flow_func, 'residual': residual}
    cut_value, partition = nx.minimum_cut(H, s, t, **kwargs)
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, G[n]) for n in reachable):
        cutset.update(((u, v) for v in nbrs if v in non_reachable))
    return cutset