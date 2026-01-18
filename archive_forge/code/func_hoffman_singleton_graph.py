from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@nx._dispatch(graphs=None)
def hoffman_singleton_graph():
    """
    Returns the Hoffman-Singleton Graph.

    The Hoffman–Singleton graph is a symmetrical undirected graph
    with 50 nodes and 175 edges.
    All indices lie in ``Z % 5``: that is, the integers mod 5 [1]_.
    It is the only regular graph of vertex degree 7, diameter 2, and girth 5.
    It is the unique (7,5)-cage graph and Moore graph, and contains many
    copies of the Petersen graph [2]_.

    Returns
    -------
    G : networkx Graph
        Hoffman–Singleton Graph with 50 nodes and 175 edges

    Notes
    -----
    Constructed from pentagon and pentagram as follows: Take five pentagons $P_h$
    and five pentagrams $Q_i$ . Join vertex $j$ of $P_h$ to vertex $h·i+j$ of $Q_i$ [3]_.

    References
    ----------
    .. [1] https://blogs.ams.org/visualinsight/2016/02/01/hoffman-singleton-graph/
    .. [2] https://mathworld.wolfram.com/Hoffman-SingletonGraph.html
    .. [3] https://en.wikipedia.org/wiki/Hoffman%E2%80%93Singleton_graph

    """
    G = nx.Graph()
    for i in range(5):
        for j in range(5):
            G.add_edge(('pentagon', i, j), ('pentagon', i, (j - 1) % 5))
            G.add_edge(('pentagon', i, j), ('pentagon', i, (j + 1) % 5))
            G.add_edge(('pentagram', i, j), ('pentagram', i, (j - 2) % 5))
            G.add_edge(('pentagram', i, j), ('pentagram', i, (j + 2) % 5))
            for k in range(5):
                G.add_edge(('pentagon', i, j), ('pentagram', k, (i * k + j) % 5))
    G = nx.convert_node_labels_to_integers(G)
    G.name = 'Hoffman-Singleton Graph'
    return G