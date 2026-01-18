import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@not_implemented_for('multigraph')
@not_implemented_for('directed')
@nx._dispatch
def one_edge_augmentation(G, avail=None, weight=None, partial=False):
    """Finds minimum weight set of edges to connect G.

    Equivalent to :func:`k_edge_augmentation` when k=1. Adding the resulting
    edges to G will make it 1-edge-connected. The solution is optimal for both
    weighted and non-weighted variants.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    partial : boolean
        If partial is True and no feasible k-edge-augmentation exists, then the
        augmenting edges minimize the number of connected components.

    Yields
    ------
    edge : tuple
        Edges in the one-augmentation of G

    Raises
    ------
    NetworkXUnfeasible
        If partial is False and no one-edge-augmentation exists.

    Notes
    -----
    Uses either :func:`unconstrained_one_edge_augmentation` or
    :func:`weighted_one_edge_augmentation` depending on whether ``avail`` is
    specified. Both algorithms are based on finding a minimum spanning tree.
    As such both algorithms find optimal solutions and run in linear time.

    See Also
    --------
    :func:`k_edge_augmentation`
    """
    if avail is None:
        return unconstrained_one_edge_augmentation(G)
    else:
        return weighted_one_edge_augmentation(G, avail=avail, weight=weight, partial=partial)