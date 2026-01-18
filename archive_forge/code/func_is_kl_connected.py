import copy
import networkx as nx
@nx._dispatch
def is_kl_connected(G, k, l, low_memory=False):
    """Returns True if and only if `G` is locally `(k, l)`-connected.

    A graph is locally `(k, l)`-connected if for each edge `(u, v)` in the
    graph there are at least `l` edge-disjoint paths of length at most `k`
    joining `u` to `v`.

    Parameters
    ----------
    G : NetworkX graph
        The graph to test for local `(k, l)`-connectedness.

    k : integer
        The maximum length of paths to consider. A higher number means a looser
        connectivity requirement.

    l : integer
        The number of edge-disjoint paths. A higher number means a stricter
        connectivity requirement.

    low_memory : bool
        If this is True, this function uses an algorithm that uses slightly
        more time but less memory.

    Returns
    -------
    bool
        Whether the graph is locally `(k, l)`-connected subgraph.

    See also
    --------
    kl_connected_subgraph

    References
    ----------
    .. [1] Chung, Fan and Linyuan Lu. "The Small World Phenomenon in Hybrid
           Power Law Graphs." *Complex Networks*. Springer Berlin Heidelberg,
           2004. 89--104.

    """
    graphOK = True
    for edge in G.edges():
        u, v = edge
        if low_memory:
            verts = {u, v}
            for i in range(k):
                [verts.update(G.neighbors(w)) for w in verts.copy()]
            G2 = G.subgraph(verts)
        else:
            G2 = copy.deepcopy(G)
        path = [u, v]
        cnt = 0
        accept = 0
        while path:
            cnt += 1
            if cnt >= l:
                accept = 1
                break
            prev = u
            for w in path:
                if w != prev:
                    G2.remove_edge(prev, w)
                    prev = w
            try:
                path = nx.shortest_path(G2, u, v)
            except nx.NetworkXNoPath:
                path = False
        if accept == 0:
            graphOK = False
            break
    return graphOK