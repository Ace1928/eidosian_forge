from math import log
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def common_neighbor_centrality(G, ebunch=None, alpha=0.8):
    """Return the CCPA score for each pair of nodes.

    Compute the Common Neighbor and Centrality based Parameterized Algorithm(CCPA)
    score of all node pairs in ebunch.

    CCPA score of `u` and `v` is defined as

    .. math::

        \\alpha \\cdot (|\\Gamma (u){\\cap }^{}\\Gamma (v)|)+(1-\\alpha )\\cdot \\frac{N}{{d}_{uv}}

    where $\\Gamma(u)$ denotes the set of neighbors of $u$, $\\Gamma(v)$ denotes the
    set of neighbors of $v$, $\\alpha$ is  parameter varies between [0,1], $N$ denotes
    total number of nodes in the Graph and ${d}_{uv}$ denotes shortest distance
    between $u$ and $v$.

    This algorithm is based on two vital properties of nodes, namely the number
    of common neighbors and their centrality. Common neighbor refers to the common
    nodes between two nodes. Centrality refers to the prestige that a node enjoys
    in a network.

    .. seealso::

        :func:`common_neighbors`

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Preferential attachment score will be computed for each pair of
        nodes given in the iterable. The pairs must be given as
        2-tuples (u, v) where u and v are nodes in the graph. If ebunch
        is None then all nonexistent edges in the graph will be used.
        Default value: None.

    alpha : Parameter defined for participation of Common Neighbor
            and Centrality Algorithm share. Values for alpha should
            normally be between 0 and 1. Default value set to 0.8
            because author found better performance at 0.8 for all the
            dataset.
            Default value: 0.8


    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Common Neighbor and Centrality based
        Parameterized Algorithm(CCPA) score.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = nx.common_neighbor_centrality(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p}")
    (0, 1) -> 3.4000000000000004
    (2, 3) -> 3.4000000000000004

    References
    ----------
    .. [1] Ahmad, I., Akhtar, M.U., Noor, S. et al.
           Missing Link Prediction using Common Neighbor and Centrality based Parameterized Algorithm.
           Sci Rep 10, 364 (2020).
           https://doi.org/10.1038/s41598-019-57304-y
    """
    if alpha == 1:

        def predict(u, v):
            if u == v:
                raise nx.NetworkXAlgorithmError('Self links are not supported')
            return sum((1 for _ in nx.common_neighbors(G, u, v)))
    else:
        spl = dict(nx.shortest_path_length(G))
        inf = float('inf')

        def predict(u, v):
            if u == v:
                raise nx.NetworkXAlgorithmError('Self links are not supported')
            path_len = spl[u].get(v, inf)
            return alpha * sum((1 for _ in nx.common_neighbors(G, u, v))) + (1 - alpha) * (G.number_of_nodes() / path_len)
    return _apply_prediction(G, predict, ebunch)