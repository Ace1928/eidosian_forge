from ``u`` to ``v`` that is not blocked. That means, there is an open
from collections import deque
from itertools import chain
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for
@not_implemented_for('undirected')
@nx._dispatchable
def find_minimal_d_separator(G, x, y, *, included=None, restricted=None):
    """Returns a minimal d-separating set between `x` and `y` if possible

    A d-separating set in a DAG is a set of nodes that blocks all
    paths between the two sets of nodes, `x` and `y`. This function
    constructs a d-separating set that is "minimal", meaning no nodes can
    be removed without it losing the d-separating property for `x` and `y`.
    If no d-separating sets exist for `x` and `y`, this returns `None`.

    In a DAG there may be more than one minimal d-separator between two
    sets of nodes. Minimal d-separators are not always unique. This function
    returns one minimal d-separator, or `None` if no d-separator exists.

    Uses the algorithm presented in [1]_. The complexity of the algorithm
    is :math:`O(m)`, where :math:`m` stands for the number of edges in
    the subgraph of G consisting of only the ancestors of `x` and `y`.
    For full details, see [1]_.

    Parameters
    ----------
    G : graph
        A networkx DAG.
    x : set | node
        A node or set of nodes in the graph.
    y : set | node
        A node or set of nodes in the graph.
    included : set | node | None
        A node or set of nodes which must be included in the found separating set,
        default is None, which means the empty set.
    restricted : set | node | None
        Restricted node or set of nodes to consider. Only these nodes can be in
        the found separating set, default is None meaning all nodes in ``G``.

    Returns
    -------
    z : set | None
        The minimal d-separating set, if at least one d-separating set exists,
        otherwise None.

    Raises
    ------
    NetworkXError
        Raises a :exc:`NetworkXError` if the input graph is not a DAG
        or if node sets `x`, `y`, and `included` are not disjoint.

    NodeNotFound
        If any of the input nodes are not found in the graph,
        a :exc:`NodeNotFound` exception is raised.

    References
    ----------
    .. [1] van der Zander, Benito, and Maciej Liśkiewicz. "Finding
        minimal d-separators in linear time and applications." In
        Uncertainty in Artificial Intelligence, pp. 637-647. PMLR, 2020.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError('graph should be directed acyclic')
    try:
        x = {x} if x in G else x
        y = {y} if y in G else y
        if included is None:
            included = set()
        elif included in G:
            included = {included}
        if restricted is None:
            restricted = set(G)
        elif restricted in G:
            restricted = {restricted}
        set_y = x | y | included | restricted
        if set_y - G.nodes:
            raise nx.NodeNotFound(f'The node(s) {set_y - G.nodes} are not found in G')
    except TypeError:
        raise nx.NodeNotFound('One of x, y, included or restricted is not a node or set of nodes in G')
    if not included <= restricted:
        raise nx.NetworkXError(f'Included nodes {included} must be in restricted nodes {restricted}')
    intersection = x & y or x & included or y & included
    if intersection:
        raise nx.NetworkXError(f'The sets x, y, included are not disjoint. Overlap: {intersection}')
    nodeset = x | y | included
    ancestors_x_y_included = nodeset.union(*[nx.ancestors(G, node) for node in nodeset])
    z_init = restricted & ancestors_x_y_included - (x | y)
    x_closure = _reachable(G, x, ancestors_x_y_included, z_init)
    if x_closure & y:
        return None
    z_updated = z_init & (x_closure | included)
    y_closure = _reachable(G, y, ancestors_x_y_included, z_updated)
    return z_updated & (y_closure | included)