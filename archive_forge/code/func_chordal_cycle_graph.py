import itertools
import networkx as nx
@nx._dispatch(graphs=None)
def chordal_cycle_graph(p, create_using=None):
    """Returns the chordal cycle graph on `p` nodes.

    The returned graph is a cycle graph on `p` nodes with chords joining each
    vertex `x` to its inverse modulo `p`. This graph is a (mildly explicit)
    3-regular expander [1]_.

    `p` *must* be a prime number.

    Parameters
    ----------
    p : a prime number

        The number of vertices in the graph. This also indicates where the
        chordal edges in the cycle will be created.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : graph
        The constructed undirected multigraph.

    Raises
    ------
    NetworkXError

        If `create_using` indicates directed or not a multigraph.

    References
    ----------

    .. [1] Theorem 4.4.2 in A. Lubotzky. "Discrete groups, expanding graphs and
           invariant measures", volume 125 of Progress in Mathematics.
           Birkhäuser Verlag, Basel, 1994.

    """
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed() or not G.is_multigraph():
        msg = '`create_using` must be an undirected multigraph.'
        raise nx.NetworkXError(msg)
    for x in range(p):
        left = (x - 1) % p
        right = (x + 1) % p
        chord = pow(x, p - 2, p) if x > 0 else 0
        for y in (left, right, chord):
            G.add_edge(x, y)
    G.graph['name'] = f'chordal_cycle_graph({p})'
    return G