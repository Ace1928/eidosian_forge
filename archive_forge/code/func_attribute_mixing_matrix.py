import networkx as nx
from networkx.algorithms.assortativity.pairs import node_attribute_xy, node_degree_xy
from networkx.utils import dict_to_numpy_array
@nx._dispatch(node_attrs='attribute')
def attribute_mixing_matrix(G, attribute, nodes=None, mapping=None, normalized=True):
    """Returns mixing matrix for attribute.

    Parameters
    ----------
    G : graph
       NetworkX graph object.

    attribute : string
       Node attribute key.

    nodes: list or iterable (optional)
        Use only nodes in container to build the matrix. The default is
        all nodes.

    mapping : dictionary, optional
       Mapping from node attribute to integer index in matrix.
       If not specified, an arbitrary ordering will be used.

    normalized : bool (default=True)
       Return counts if False or probabilities if True.

    Returns
    -------
    m: numpy array
       Counts or joint probability of occurrence of attribute pairs.

    Notes
    -----
    If each node has a unique attribute value, the unnormalized mixing matrix
    will be equal to the adjacency matrix. To get a denser mixing matrix,
    the rounding can be performed to form groups of nodes with equal values.
    For example, the exact height of persons in cm (180.79155222, 163.9080892,
    163.30095355, 167.99016217, 168.21590163, ...) can be rounded to (180, 163,
    163, 168, 168, ...).

    Definitions of attribute mixing matrix vary on whether the matrix
    should include rows for attribute values that don't arise. Here we
    do not include such empty-rows. But you can force them to appear
    by inputting a `mapping` that includes those values.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> gender = {0: 'male', 1: 'female', 2: 'female'}
    >>> nx.set_node_attributes(G, gender, 'gender')
    >>> mapping = {'male': 0, 'female': 1}
    >>> mix_mat = nx.attribute_mixing_matrix(G, 'gender', mapping=mapping)
    >>> # mixing from male nodes to female nodes
    >>> mix_mat[mapping['male'], mapping['female']]
    0.25
    """
    d = attribute_mixing_dict(G, attribute, nodes)
    a = dict_to_numpy_array(d, mapping=mapping)
    if normalized:
        a = a / a.sum()
    return a