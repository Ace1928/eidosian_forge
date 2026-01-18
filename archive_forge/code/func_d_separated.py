from collections import deque
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def d_separated(G, x, y, z):
    """
    Return whether node sets ``x`` and ``y`` are d-separated by ``z``.

    Parameters
    ----------
    G : graph
        A NetworkX DAG.

    x : set
        First set of nodes in ``G``.

    y : set
        Second set of nodes in ``G``.

    z : set
        Set of conditioning nodes in ``G``. Can be empty set.

    Returns
    -------
    b : bool
        A boolean that is true if ``x`` is d-separated from ``y`` given ``z`` in ``G``.

    Raises
    ------
    NetworkXError
        The *d-separation* test is commonly used with directed
        graphical models which are acyclic.  Accordingly, the algorithm
        raises a :exc:`NetworkXError` if the input graph is not a DAG.

    NodeNotFound
        If any of the input nodes are not found in the graph,
        a :exc:`NodeNotFound` exception is raised.

    Notes
    -----
    A d-separating set in a DAG is a set of nodes that
    blocks all paths between the two sets. Nodes in `z`
    block a path if they are part of the path and are not a collider,
    or a descendant of a collider. A collider structure along a path
    is ``... -> c <- ...`` where ``c`` is the collider node.

    https://en.wikipedia.org/wiki/Bayesian_network#d-separation
    """
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError('graph should be directed acyclic')
    union_xyz = x.union(y).union(z)
    if any((n not in G.nodes for n in union_xyz)):
        raise nx.NodeNotFound('one or more specified nodes not found in the graph')
    G_copy = G.copy()
    leaves = deque([n for n in G_copy.nodes if G_copy.out_degree[n] == 0])
    while len(leaves) > 0:
        leaf = leaves.popleft()
        if leaf not in union_xyz:
            for p in G_copy.predecessors(leaf):
                if G_copy.out_degree[p] == 1:
                    leaves.append(p)
            G_copy.remove_node(leaf)
    edges_to_remove = list(G_copy.out_edges(z))
    G_copy.remove_edges_from(edges_to_remove)
    disjoint_set = UnionFind(G_copy.nodes())
    for component in nx.weakly_connected_components(G_copy):
        disjoint_set.union(*component)
    disjoint_set.union(*x)
    disjoint_set.union(*y)
    if x and y and (disjoint_set[next(iter(x))] == disjoint_set[next(iter(y))]):
        return False
    else:
        return True