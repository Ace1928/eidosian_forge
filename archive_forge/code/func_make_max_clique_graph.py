from collections import defaultdict, deque
from itertools import chain, combinations, islice
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch
def make_max_clique_graph(G, create_using=None):
    """Returns the maximal clique graph of the given graph.

    The nodes of the maximal clique graph of `G` are the cliques of
    `G` and an edge joins two cliques if the cliques are not disjoint.

    Parameters
    ----------
    G : NetworkX graph

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX graph
        A graph whose nodes are the cliques of `G` and whose edges
        join two cliques if they are not disjoint.

    Notes
    -----
    This function behaves like the following code::

        import networkx as nx
        G = nx.make_clique_bipartite(G)
        cliques = [v for v in G.nodes() if G.nodes[v]['bipartite'] == 0]
        G = nx.bipartite.projected_graph(G, cliques)
        G = nx.relabel_nodes(G, {-v: v - 1 for v in G})

    It should be faster, though, since it skips all the intermediate
    steps.

    """
    if create_using is None:
        B = G.__class__()
    else:
        B = nx.empty_graph(0, create_using)
    cliques = list(enumerate((set(c) for c in find_cliques(G))))
    B.add_nodes_from((i for i, c in cliques))
    clique_pairs = combinations(cliques, 2)
    B.add_edges_from(((i, j) for (i, c1), (j, c2) in clique_pairs if c1 & c2))
    return B