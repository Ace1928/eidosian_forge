import networkx as nx
from networkx.algorithms.assortativity.pairs import node_attribute_xy, node_degree_xy
from networkx.utils import dict_to_numpy_array
@nx._dispatch(edge_attrs='weight')
def degree_mixing_matrix(G, x='out', y='in', weight=None, nodes=None, normalized=True, mapping=None):
    """Returns mixing matrix for attribute.

    Parameters
    ----------
    G : graph
       NetworkX graph object.

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    nodes: list or iterable (optional)
        Build the matrix using only nodes in container.
        The default is all nodes.

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    normalized : bool (default=True)
       Return counts if False or probabilities if True.

    mapping : dictionary, optional
       Mapping from node degree to integer index in matrix.
       If not specified, an arbitrary ordering will be used.

    Returns
    -------
    m: numpy array
       Counts, or joint probability, of occurrence of node degree.

    Notes
    -----
    Definitions of degree mixing matrix vary on whether the matrix
    should include rows for degree values that don't arise. Here we
    do not include such empty-rows. But you can force them to appear
    by inputting a `mapping` that includes those values. See examples.

    Examples
    --------
    >>> G = nx.star_graph(3)
    >>> mix_mat = nx.degree_mixing_matrix(G)
    >>> mix_mat[0, 1]  # mixing from node degree 1 to node degree 3
    0.5

    If you want every possible degree to appear as a row, even if no nodes
    have that degree, use `mapping` as follows,

    >>> max_degree = max(deg for n, deg in G.degree)
    >>> mapping = {x: x for x in range(max_degree + 1)} # identity mapping
    >>> mix_mat = nx.degree_mixing_matrix(G, mapping=mapping)
    >>> mix_mat[3, 1]  # mixing from node degree 3 to node degree 1
    0.5
    """
    d = degree_mixing_dict(G, x=x, y=y, nodes=nodes, weight=weight)
    a = dict_to_numpy_array(d, mapping=mapping)
    if normalized:
        a = a / a.sum()
    return a