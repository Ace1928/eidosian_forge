from itertools import product
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(graphs=_G_H)
def corona_product(G, H):
    """Returns the Corona product of G and H.

    The corona product of $G$ and $H$ is the graph $C = G \\circ H$ obtained by
    taking one copy of $G$, called the center graph, $|V(G)|$ copies of $H$,
    called the outer graph, and making the $i$-th vertex of $G$ adjacent to
    every vertex of the $i$-th copy of $H$, where $1 ≤ i ≤ |V(G)|$.

    Parameters
    ----------
    G, H: NetworkX graphs
        The graphs to take the carona product of.
        `G` is the center graph and `H` is the outer graph

    Returns
    -------
    C: NetworkX graph
        The Corona product of G and H.

    Raises
    ------
    NetworkXError
        If G and H are not both directed or both undirected.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> H = nx.path_graph(2)
    >>> C = nx.corona_product(G, H)
    >>> list(C)
    [0, 1, 2, 3, (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    >>> print(C)
    Graph with 12 nodes and 16 edges

    References
    ----------
    [1] M. Tavakoli, F. Rahbarnia, and A. R. Ashrafi,
        "Studying the corona product of graphs under some graph invariants,"
        Transactions on Combinatorics, vol. 3, no. 3, pp. 43–49, Sep. 2014,
        doi: 10.22108/toc.2014.5542.
    [2] A. Faraji, "Corona Product in Graph Theory," Ali Faraji, May 11, 2021.
        https://blog.alifaraji.ir/math/graph-theory/corona-product.html (accessed Dec. 07, 2021).
    """
    GH = _init_product_graph(G, H)
    GH.add_nodes_from(G)
    GH.add_edges_from(G.edges)
    for G_node in G:
        GH.add_nodes_from(((G_node, v) for v in H))
        GH.add_edges_from((((G_node, e0), (G_node, e1), d) for e0, e1, d in H.edges.data()))
        GH.add_edges_from(((G_node, (G_node, H_node)) for H_node in H))
    return GH