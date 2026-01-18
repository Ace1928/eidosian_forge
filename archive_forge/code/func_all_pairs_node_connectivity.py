import itertools
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import (
from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity
@nx._dispatch
def all_pairs_node_connectivity(G, nbunch=None, flow_func=None):
    """Compute node connectivity between all pairs of nodes of G.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    nbunch: container
        Container of nodes. If provided node connectivity will be computed
        only over pairs of nodes in nbunch.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The
        choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    Returns
    -------
    all_pairs : dict
        A dictionary with node connectivity between all pairs of nodes
        in G, or in nbunch if provided.

    See also
    --------
    :meth:`local_node_connectivity`
    :meth:`edge_connectivity`
    :meth:`local_edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    """
    if nbunch is None:
        nbunch = G
    else:
        nbunch = set(nbunch)
    directed = G.is_directed()
    if directed:
        iter_func = itertools.permutations
    else:
        iter_func = itertools.combinations
    all_pairs = {n: {} for n in nbunch}
    H = build_auxiliary_node_connectivity(G)
    mapping = H.graph['mapping']
    R = build_residual_network(H, 'capacity')
    kwargs = {'flow_func': flow_func, 'auxiliary': H, 'residual': R}
    for u, v in iter_func(nbunch, 2):
        K = local_node_connectivity(G, u, v, **kwargs)
        all_pairs[u][v] = K
        if not directed:
            all_pairs[v][u] = K
    return all_pairs