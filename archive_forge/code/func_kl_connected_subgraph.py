import copy
import networkx as nx
@nx._dispatch
def kl_connected_subgraph(G, k, l, low_memory=False, same_as_graph=False):
    """Returns the maximum locally `(k, l)`-connected subgraph of `G`.

    A graph is locally `(k, l)`-connected if for each edge `(u, v)` in the
    graph there are at least `l` edge-disjoint paths of length at most `k`
    joining `u` to `v`.

    Parameters
    ----------
    G : NetworkX graph
        The graph in which to find a maximum locally `(k, l)`-connected
        subgraph.

    k : integer
        The maximum length of paths to consider. A higher number means a looser
        connectivity requirement.

    l : integer
        The number of edge-disjoint paths. A higher number means a stricter
        connectivity requirement.

    low_memory : bool
        If this is True, this function uses an algorithm that uses slightly
        more time but less memory.

    same_as_graph : bool
        If True then return a tuple of the form `(H, is_same)`,
        where `H` is the maximum locally `(k, l)`-connected subgraph and
        `is_same` is a Boolean representing whether `G` is locally `(k,
        l)`-connected (and hence, whether `H` is simply a copy of the input
        graph `G`).

    Returns
    -------
    NetworkX graph or two-tuple
        If `same_as_graph` is True, then this function returns a
        two-tuple as described above. Otherwise, it returns only the maximum
        locally `(k, l)`-connected subgraph.

    See also
    --------
    is_kl_connected

    References
    ----------
    .. [1] Chung, Fan and Linyuan Lu. "The Small World Phenomenon in Hybrid
           Power Law Graphs." *Complex Networks*. Springer Berlin Heidelberg,
           2004. 89--104.

    """
    H = copy.deepcopy(G)
    graphOK = True
    deleted_some = True
    while deleted_some:
        deleted_some = False
        for edge in list(H.edges()):
            u, v = edge
            if low_memory:
                verts = {u, v}
                for i in range(k):
                    for w in verts.copy():
                        verts.update(G[w])
                G2 = G.subgraph(verts).copy()
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
                    if prev != w:
                        G2.remove_edge(prev, w)
                        prev = w
                try:
                    path = nx.shortest_path(G2, u, v)
                except nx.NetworkXNoPath:
                    path = False
            if accept == 0:
                H.remove_edge(u, v)
                deleted_some = True
                if graphOK:
                    graphOK = False
    if same_as_graph:
        return (H, graphOK)
    return H