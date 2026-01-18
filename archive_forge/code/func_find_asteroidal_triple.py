import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def find_asteroidal_triple(G):
    """Find an asteroidal triple in the given graph.

    An asteroidal triple is a triple of non-adjacent vertices such that
    there exists a path between any two of them which avoids the closed
    neighborhood of the third. It checks all independent triples of vertices
    and whether they are an asteroidal triple or not. This is done with the
    help of a data structure called a component structure.
    A component structure encodes information about which vertices belongs to
    the same connected component when the closed neighborhood of a given vertex
    is removed from the graph. The algorithm used to check is the trivial
    one, outlined in [1]_, which has a runtime of
    :math:`O(|V||\\overline{E} + |V||E|)`, where the second term is the
    creation of the component structure.

    Parameters
    ----------
    G : NetworkX Graph
        The graph to check whether is AT-free or not

    Returns
    -------
    list or None
        An asteroidal triple is returned as a list of nodes. If no asteroidal
        triple exists, i.e. the graph is AT-free, then None is returned.
        The returned value depends on the certificate parameter. The default
        option is a bool which is True if the graph is AT-free, i.e. the
        given graph contains no asteroidal triples, and False otherwise, i.e.
        if the graph contains at least one asteroidal triple.

    Notes
    -----
    The component structure and the algorithm is described in [1]_. The current
    implementation implements the trivial algorithm for simple graphs.

    References
    ----------
    .. [1] Ekkehard Köhler,
       "Recognizing Graphs without asteroidal triples",
       Journal of Discrete Algorithms 2, pages 439-452, 2004.
       https://www.sciencedirect.com/science/article/pii/S157086670400019X
    """
    V = set(G.nodes)
    if len(V) < 6:
        return None
    component_structure = create_component_structure(G)
    E_complement = set(nx.complement(G).edges)
    for e in E_complement:
        u = e[0]
        v = e[1]
        u_neighborhood = set(G[u]).union([u])
        v_neighborhood = set(G[v]).union([v])
        union_of_neighborhoods = u_neighborhood.union(v_neighborhood)
        for w in V - union_of_neighborhoods:
            if component_structure[u][v] == component_structure[u][w] and component_structure[v][u] == component_structure[v][w] and (component_structure[w][u] == component_structure[w][v]):
                return [u, v, w]
    return None