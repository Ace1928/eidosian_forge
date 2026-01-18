import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@not_implemented_for('multigraph')
@not_implemented_for('directed')
@nx._dispatch
def bridge_augmentation(G, avail=None, weight=None):
    """Finds the a set of edges that bridge connects G.

    Equivalent to :func:`k_edge_augmentation` when k=2, and partial=False.
    Adding the resulting edges to G will make it 2-edge-connected.  If no
    constraints are specified the returned set of edges is minimum an optimal,
    otherwise the solution is approximated.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    Yields
    ------
    edge : tuple
        Edges in the bridge-augmentation of G

    Raises
    ------
    NetworkXUnfeasible
        If no bridge-augmentation exists.

    Notes
    -----
    If there are no constraints the solution can be computed in linear time
    using :func:`unconstrained_bridge_augmentation`. Otherwise, the problem
    becomes NP-hard and is the solution is approximated by
    :func:`weighted_bridge_augmentation`.

    See Also
    --------
    :func:`k_edge_augmentation`
    """
    if G.number_of_nodes() < 3:
        raise nx.NetworkXUnfeasible('impossible to bridge connect less than 3 nodes')
    if avail is None:
        return unconstrained_bridge_augmentation(G)
    else:
        return weighted_bridge_augmentation(G, avail, weight=weight)