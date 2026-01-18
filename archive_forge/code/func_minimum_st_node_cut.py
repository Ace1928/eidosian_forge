import itertools
import networkx as nx
from networkx.algorithms.flow import build_residual_network, edmonds_karp
from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity
@nx._dispatch(graphs={'G': 0, 'auxiliary?': 4, 'residual?': 5}, preserve_edge_attrs={'residual': {'capacity': float('inf')}}, preserve_node_attrs={'auxiliary': {'id': None}}, preserve_graph_attrs={'auxiliary', 'residual'})
def minimum_st_node_cut(G, s, t, flow_func=None, auxiliary=None, residual=None):
    """Returns a set of nodes of minimum cardinality that disconnect source
    from target in G.

    This function returns the set of nodes of minimum cardinality that,
    if removed, would destroy all paths among source and target in G.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node.

    t : node
        Target node.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The choice
        of the default function may change from version to version and
        should not be relied on. Default value: None.

    auxiliary : NetworkX DiGraph
        Auxiliary digraph to compute flow based node connectivity. It has
        to have a graph attribute called mapping with a dictionary mapping
        node names in G and in the auxiliary digraph. If provided
        it will be reused instead of recreated. Default value: None.

    residual : NetworkX DiGraph
        Residual network to compute maximum flow. If provided it will be
        reused instead of recreated. Default value: None.

    Returns
    -------
    cutset : set
        Set of nodes that, if removed, would destroy all paths between
        source and target in G.

    Examples
    --------
    This function is not imported in the base NetworkX namespace, so you
    have to explicitly import it from the connectivity package:

    >>> from networkx.algorithms.connectivity import minimum_st_node_cut

    We use in this example the platonic icosahedral graph, which has node
    connectivity 5.

    >>> G = nx.icosahedral_graph()
    >>> len(minimum_st_node_cut(G, 0, 6))
    5

    If you need to compute local st cuts between several pairs of
    nodes in the same graph, it is recommended that you reuse the
    data structures that NetworkX uses in the computation: the
    auxiliary digraph for node connectivity and node cuts, and the
    residual network for the underlying maximum flow computation.

    Example of how to compute local st node cuts reusing the data
    structures:

    >>> # You also have to explicitly import the function for
    >>> # building the auxiliary digraph from the connectivity package
    >>> from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
    >>> H = build_auxiliary_node_connectivity(G)
    >>> # And the function for building the residual network from the
    >>> # flow package
    >>> from networkx.algorithms.flow import build_residual_network
    >>> # Note that the auxiliary digraph has an edge attribute named capacity
    >>> R = build_residual_network(H, "capacity")
    >>> # Reuse the auxiliary digraph and the residual network by passing them
    >>> # as parameters
    >>> len(minimum_st_node_cut(G, 0, 6, auxiliary=H, residual=R))
    5

    You can also use alternative flow algorithms for computing minimum st
    node cuts. For instance, in dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better than
    the default :meth:`edmonds_karp` which is faster for sparse
    networks with highly skewed degree distributions. Alternative flow
    functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> len(minimum_st_node_cut(G, 0, 6, flow_func=shortest_augmenting_path))
    5

    Notes
    -----
    This is a flow based implementation of minimum node cut. The algorithm
    is based in solving a number of maximum flow computations to determine
    the capacity of the minimum cut on an auxiliary directed network that
    corresponds to the minimum node cut of G. It handles both directed
    and undirected graphs. This implementation is based on algorithm 11
    in [1]_.

    See also
    --------
    :meth:`minimum_node_cut`
    :meth:`minimum_edge_cut`
    :meth:`stoer_wagner`
    :meth:`node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """
    if auxiliary is None:
        H = build_auxiliary_node_connectivity(G)
    else:
        H = auxiliary
    mapping = H.graph.get('mapping', None)
    if mapping is None:
        raise nx.NetworkXError('Invalid auxiliary digraph.')
    if G.has_edge(s, t) or G.has_edge(t, s):
        return {}
    kwargs = {'flow_func': flow_func, 'residual': residual, 'auxiliary': H}
    edge_cut = minimum_st_edge_cut(H, f'{mapping[s]}B', f'{mapping[t]}A', **kwargs)
    node_cut = {H.nodes[node]['id'] for edge in edge_cut for node in edge}
    return node_cut - {s, t}