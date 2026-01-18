import networkx as nx
from networkx.algorithms.assortativity.pairs import node_attribute_xy, node_degree_xy
from networkx.utils import dict_to_numpy_array
@nx._dispatch(node_attrs='attribute')
def attribute_mixing_dict(G, attribute, nodes=None, normalized=False):
    """Returns dictionary representation of mixing matrix for attribute.

    Parameters
    ----------
    G : graph
       NetworkX graph object.

    attribute : string
       Node attribute key.

    nodes: list or iterable (optional)
        Unse nodes in container to build the dict. The default is all nodes.

    normalized : bool (default=False)
       Return counts if False or probabilities if True.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_nodes_from([0, 1], color="red")
    >>> G.add_nodes_from([2, 3], color="blue")
    >>> G.add_edge(1, 3)
    >>> d = nx.attribute_mixing_dict(G, "color")
    >>> print(d["red"]["blue"])
    1
    >>> print(d["blue"]["red"])  # d symmetric for undirected graphs
    1

    Returns
    -------
    d : dictionary
       Counts or joint probability of occurrence of attribute pairs.
    """
    xy_iter = node_attribute_xy(G, attribute, nodes)
    return mixing_dict(xy_iter, normalized=normalized)