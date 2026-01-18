from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from math import isnan
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for, py_random_state
@nx._dispatchable(edge_attrs='weight')
def number_of_spanning_trees(G, *, root=None, weight=None):
    """Returns the number of spanning trees in `G`.

    A spanning tree for an undirected graph is a tree that connects
    all nodes in the graph. For a directed graph, the analog of a
    spanning tree is called a (spanning) arborescence. The arborescence
    includes a unique directed path from the `root` node to each other node.
    The graph must be weakly connected, and the root must be a node
    that includes all nodes as successors [3]_. Note that to avoid
    discussing sink-roots and reverse-arborescences, we have reversed
    the edge orientation from [3]_ and use the in-degree laplacian.

    This function (when `weight` is `None`) returns the number of
    spanning trees for an undirected graph and the number of
    arborescences from a single root node for a directed graph.
    When `weight` is the name of an edge attribute which holds the
    weight value of each edge, the function returns the sum over
    all trees of the multiplicative weight of each tree. That is,
    the weight of the tree is the product of its edge weights.

    Kirchoff's Tree Matrix Theorem states that any cofactor of the
    Laplacian matrix of a graph is the number of spanning trees in the
    graph. (Here we use cofactors for a diagonal entry so that the
    cofactor becomes the determinant of the matrix with one row
    and its matching column removed.) For a weighted Laplacian matrix,
    the cofactor is the sum across all spanning trees of the
    multiplicative weight of each tree. That is, the weight of each
    tree is the product of its edge weights. The theorem is also
    known as Kirchhoff's theorem [1]_ and the Matrix-Tree theorem [2]_.

    For directed graphs, a similar theorem (Tutte's Theorem) holds with
    the cofactor chosen to be the one with row and column removed that
    correspond to the root. The cofactor is the number of arborescences
    with the specified node as root. And the weighted version gives the
    sum of the arborescence weights with root `root`. The arborescence
    weight is the product of its edge weights.

    Parameters
    ----------
    G : NetworkX graph

    root : node
       A node in the directed graph `G` that has all nodes as descendants.
       (This is ignored for undirected graphs.)

    weight : string or None, optional (default=None)
        The name of the edge attribute holding the edge weight.
        If `None`, then each edge is assumed to have a weight of 1.

    Returns
    -------
    Number
        Undirected graphs:
            The number of spanning trees of the graph `G`.
            Or the sum of all spanning tree weights of the graph `G`
            where the weight of a tree is the product of its edge weights.
        Directed graphs:
            The number of arborescences of `G` rooted at node `root`.
            Or the sum of all arborescence weights of the graph `G` with
            specified root where the weight of an arborescence is the product
            of its edge weights.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` does not contain any nodes.

    NetworkXError
        If the graph `G` is directed and the root node
        is not specified or is not in G.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> round(nx.number_of_spanning_trees(G))
    125

    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=2)
    >>> G.add_edge(1, 3, weight=1)
    >>> G.add_edge(2, 3, weight=1)
    >>> round(nx.number_of_spanning_trees(G, weight="weight"))
    5

    Notes
    -----
    Self-loops are excluded. Multi-edges are contracted in one edge
    equal to the sum of the weights.

    References
    ----------
    .. [1] Wikipedia
       "Kirchhoff's theorem."
       https://en.wikipedia.org/wiki/Kirchhoff%27s_theorem
    .. [2] Kirchhoff, G. R.
        Über die Auflösung der Gleichungen, auf welche man
        bei der Untersuchung der linearen Vertheilung
        Galvanischer Ströme geführt wird
        Annalen der Physik und Chemie, vol. 72, pp. 497-508, 1847.
    .. [3] Margoliash, J.
        "Matrix-Tree Theorem for Directed Graphs"
        https://www.math.uchicago.edu/~may/VIGRE/VIGRE2010/REUPapers/Margoliash.pdf
    """
    import numpy as np
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('Graph G must contain at least one node.')
    if not nx.is_directed(G):
        if not nx.is_connected(G):
            return 0
        G_laplacian = nx.laplacian_matrix(G, weight=weight).toarray()
        return float(np.linalg.det(G_laplacian[1:, 1:]))
    if root is None:
        raise nx.NetworkXError('Input `root` must be provided when G is directed')
    if root not in G:
        raise nx.NetworkXError('The node root is not in the graph G.')
    if not nx.is_weakly_connected(G):
        return 0
    nodelist = [root] + [n for n in G if n != root]
    A = nx.adjacency_matrix(G, nodelist=nodelist, weight=weight)
    D = np.diag(A.sum(axis=0))
    G_laplacian = D - A
    return float(np.linalg.det(G_laplacian[1:, 1:]))