from itertools import product
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('multigraph')
@nx._dispatch(graphs=_G_H)
def rooted_product(G, H, root):
    """Return the rooted product of graphs G and H rooted at root in H.

    A new graph is constructed representing the rooted product of
    the inputted graphs, G and H, with a root in H.
    A rooted product duplicates H for each nodes in G with the root
    of H corresponding to the node in G. Nodes are renamed as the direct
    product of G and H. The result is a subgraph of the cartesian product.

    Parameters
    ----------
    G,H : graph
       A NetworkX graph
    root : node
       A node in H

    Returns
    -------
    R : The rooted product of G and H with a specified root in H

    Notes
    -----
    The nodes of R are the Cartesian Product of the nodes of G and H.
    The nodes of G and H are not relabeled.
    """
    if root not in H:
        raise nx.NetworkXError('root must be a vertex in H')
    R = nx.Graph()
    R.add_nodes_from(product(G, H))
    R.add_edges_from((((e[0], root), (e[1], root)) for e in G.edges()))
    R.add_edges_from((((g, e[0]), (g, e[1])) for g in G for e in H.edges()))
    return R