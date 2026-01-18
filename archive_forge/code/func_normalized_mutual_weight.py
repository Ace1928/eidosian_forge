import networkx as nx
@nx._dispatch(edge_attrs='weight')
def normalized_mutual_weight(G, u, v, norm=sum, weight=None):
    """Returns normalized mutual weight of the edges from `u` to `v`
    with respect to the mutual weights of the neighbors of `u` in `G`.

    `norm` specifies how the normalization factor is computed. It must
    be a function that takes a single argument and returns a number.
    The argument will be an iterable of mutual weights
    of pairs ``(u, w)``, where ``w`` ranges over each (in- and
    out-)neighbor of ``u``. Commons values for `normalization` are
    ``sum`` and ``max``.

    `weight` can be ``None`` or a string, if None, all edge weights
    are considered equal. Otherwise holds the name of the edge
    attribute used as weight.

    """
    scale = norm((mutual_weight(G, u, w, weight) for w in set(nx.all_neighbors(G, u))))
    return 0 if scale == 0 else mutual_weight(G, u, v, weight) / scale