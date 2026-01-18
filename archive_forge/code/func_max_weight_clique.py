from collections import defaultdict, deque
from itertools import chain, combinations, islice
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@nx._dispatch(node_attrs='weight')
def max_weight_clique(G, weight='weight'):
    """Find a maximum weight clique in G.

    A *clique* in a graph is a set of nodes such that every two distinct nodes
    are adjacent.  The *weight* of a clique is the sum of the weights of its
    nodes.  A *maximum weight clique* of graph G is a clique C in G such that
    no clique in G has weight greater than the weight of C.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph
    weight : string or None, optional (default='weight')
        The node attribute that holds the integer value used as a weight.
        If None, then each node has weight 1.

    Returns
    -------
    clique : list
        the nodes of a maximum weight clique
    weight : int
        the weight of a maximum weight clique

    Notes
    -----
    The implementation is recursive, and therefore it may run into recursion
    depth issues if G contains a clique whose number of nodes is close to the
    recursion depth limit.

    At each search node, the algorithm greedily constructs a weighted
    independent set cover of part of the graph in order to find a small set of
    nodes on which to branch.  The algorithm is very similar to the algorithm
    of Tavares et al. [1]_, other than the fact that the NetworkX version does
    not use bitsets.  This style of algorithm for maximum weight clique (and
    maximum weight independent set, which is the same problem but on the
    complement graph) has a decades-long history.  See Algorithm B of Warren
    and Hicks [2]_ and the references in that paper.

    References
    ----------
    .. [1] Tavares, W.A., Neto, M.B.C., Rodrigues, C.D., Michelon, P.: Um
           algoritmo de branch and bound para o problema da clique máxima
           ponderada.  Proceedings of XLVII SBPO 1 (2015).

    .. [2] Warren, Jeffrey S, Hicks, Illya V.: Combinatorial Branch-and-Bound
           for the Maximum Weight Independent Set Problem.  Technical Report,
           Texas A&M University (2016).
    """
    mwc = MaxWeightClique(G, weight)
    mwc.find_max_weight_clique()
    return (mwc.incumbent_nodes, mwc.incumbent_weight)