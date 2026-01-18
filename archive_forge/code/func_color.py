import networkx as nx
from networkx.algorithms.components import connected_components
from networkx.exception import AmbiguousSolution
@nx._dispatch
def color(G):
    """Returns a two-coloring of the graph.

    Raises an exception if the graph is not bipartite.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    color : dictionary
        A dictionary keyed by node with a 1 or 0 as data for each node color.

    Raises
    ------
    NetworkXError
        If the graph is not two-colorable.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> c = bipartite.color(G)
    >>> print(c)
    {0: 1, 1: 0, 2: 1, 3: 0}

    You can use this to set a node attribute indicating the bipartite set:

    >>> nx.set_node_attributes(G, c, "bipartite")
    >>> print(G.nodes[0]["bipartite"])
    1
    >>> print(G.nodes[1]["bipartite"])
    0
    """
    if G.is_directed():
        import itertools

        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors(v), G.successors(v)])
    else:
        neighbors = G.neighbors
    color = {}
    for n in G:
        if n in color or len(G[n]) == 0:
            continue
        queue = [n]
        color[n] = 1
        while queue:
            v = queue.pop()
            c = 1 - color[v]
            for w in neighbors(v):
                if w in color:
                    if color[w] == color[v]:
                        raise nx.NetworkXError('Graph is not bipartite.')
                else:
                    color[w] = c
                    queue.append(w)
    color.update(dict.fromkeys(nx.isolates(G), 0))
    return color