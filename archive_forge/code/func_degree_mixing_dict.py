import networkx as nx
from networkx.algorithms.assortativity.pairs import node_attribute_xy, node_degree_xy
from networkx.utils import dict_to_numpy_array
@nx._dispatch(edge_attrs='weight')
def degree_mixing_dict(G, x='out', y='in', weight=None, nodes=None, normalized=False):
    """Returns dictionary representation of mixing matrix for degree.

    Parameters
    ----------
    G : graph
        NetworkX graph object.

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    normalized : bool (default=False)
        Return counts if False or probabilities if True.

    Returns
    -------
    d: dictionary
       Counts or joint probability of occurrence of degree pairs.
    """
    xy_iter = node_degree_xy(G, x=x, y=y, nodes=nodes, weight=weight)
    return mixing_dict(xy_iter, normalized=normalized)