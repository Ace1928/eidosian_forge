from itertools import chain, combinations, permutations, product
import networkx as nx
from networkx import density
from networkx.exception import NetworkXException
from networkx.utils import arbitrary_element
@nx._dispatch(preserve_all_attrs=True)
def contracted_nodes(G, u, v, self_loops=True, copy=True):
    """Returns the graph that results from contracting `u` and `v`.

    Node contraction identifies the two nodes as a single node incident to any
    edge that was incident to the original two nodes.

    Parameters
    ----------
    G : NetworkX graph
        The graph whose nodes will be contracted.

    u, v : nodes
        Must be nodes in `G`.

    self_loops : Boolean
        If this is True, any edges joining `u` and `v` in `G` become
        self-loops on the new node in the returned graph.

    copy : Boolean
        If this is True (default True), make a copy of
        `G` and return that instead of directly changing `G`.


    Returns
    -------
    Networkx graph
        If Copy is True,
        A new graph object of the same type as `G` (leaving `G` unmodified)
        with `u` and `v` identified in a single node. The right node `v`
        will be merged into the node `u`, so only `u` will appear in the
        returned graph.
        If copy is False,
        Modifies `G` with `u` and `v` identified in a single node.
        The right node `v` will be merged into the node `u`, so
        only `u` will appear in the returned graph.

    Notes
    -----
    For multigraphs, the edge keys for the realigned edges may
    not be the same as the edge keys for the old edges. This is
    natural because edge keys are unique only within each pair of nodes.

    For non-multigraphs where `u` and `v` are adjacent to a third node
    `w`, the edge (`v`, `w`) will be contracted into the edge (`u`,
    `w`) with its attributes stored into a "contraction" attribute.

    This function is also available as `identified_nodes`.

    Examples
    --------
    Contracting two nonadjacent nodes of the cycle graph on four nodes `C_4`
    yields the path graph (ignoring parallel edges):

    >>> G = nx.cycle_graph(4)
    >>> M = nx.contracted_nodes(G, 1, 3)
    >>> P3 = nx.path_graph(3)
    >>> nx.is_isomorphic(M, P3)
    True

    >>> G = nx.MultiGraph(P3)
    >>> M = nx.contracted_nodes(G, 0, 2)
    >>> M.edges
    MultiEdgeView([(0, 1, 0), (0, 1, 1)])

    >>> G = nx.Graph([(1, 2), (2, 2)])
    >>> H = nx.contracted_nodes(G, 1, 2, self_loops=False)
    >>> list(H.nodes())
    [1]
    >>> list(H.edges())
    [(1, 1)]

    In a ``MultiDiGraph`` with a self loop, the in and out edges will
    be treated separately as edges, so while contracting a node which
    has a self loop the contraction will add multiple edges:

    >>> G = nx.MultiDiGraph([(1, 2), (2, 2)])
    >>> H = nx.contracted_nodes(G, 1, 2)
    >>> list(H.edges())  # edge 1->2, 2->2, 2<-2 from the original Graph G
    [(1, 1), (1, 1), (1, 1)]
    >>> H = nx.contracted_nodes(G, 1, 2, self_loops=False)
    >>> list(H.edges())  # edge 2->2, 2<-2 from the original Graph G
    [(1, 1), (1, 1)]

    See Also
    --------
    contracted_edge
    quotient_graph

    """
    if copy:
        H = G.copy()
    else:
        H = G
    if H.is_directed():
        edges_to_remap = chain(G.in_edges(v, data=True), G.out_edges(v, data=True))
    else:
        edges_to_remap = G.edges(v, data=True)
    if not copy:
        edges_to_remap = list(edges_to_remap)
    v_data = H.nodes[v]
    H.remove_node(v)
    for prev_w, prev_x, d in edges_to_remap:
        w = prev_w if prev_w != v else u
        x = prev_x if prev_x != v else u
        if {prev_w, prev_x} == {u, v} and (not self_loops):
            continue
        if not H.has_edge(w, x) or G.is_multigraph():
            H.add_edge(w, x, **d)
        elif 'contraction' in H.edges[w, x]:
            H.edges[w, x]['contraction'][prev_w, prev_x] = d
        else:
            H.edges[w, x]['contraction'] = {(prev_w, prev_x): d}
    if 'contraction' in H.nodes[u]:
        H.nodes[u]['contraction'][v] = v_data
    else:
        H.nodes[u]['contraction'] = {v: v_data}
    return H