import networkx as nx
@nx._dispatch(edge_attrs='weight')
def mutual_weight(G, u, v, weight=None):
    """Returns the sum of the weights of the edge from `u` to `v` and
    the edge from `v` to `u` in `G`.

    `weight` is the edge data key that represents the edge weight. If
    the specified key is `None` or is not in the edge data for an edge,
    that edge is assumed to have weight 1.

    Pre-conditions: `u` and `v` must both be in `G`.

    """
    try:
        a_uv = G[u][v].get(weight, 1)
    except KeyError:
        a_uv = 0
    try:
        a_vu = G[v][u].get(weight, 1)
    except KeyError:
        a_vu = 0
    return a_uv + a_vu