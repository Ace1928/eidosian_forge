from math import log
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def preferential_attachment(G, ebunch=None):
    """Compute the preferential attachment score of all node pairs in ebunch.

    Preferential attachment score of `u` and `v` is defined as

    .. math::

        |\\Gamma(u)| |\\Gamma(v)|

    where $\\Gamma(u)$ denotes the set of neighbors of $u$.

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

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their preferential attachment score.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = nx.preferential_attachment(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p}")
    (0, 1) -> 16
    (2, 3) -> 16

    References
    ----------
    .. [1] D. Liben-Nowell, J. Kleinberg.
           The Link Prediction Problem for Social Networks (2004).
           http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    """

    def predict(u, v):
        return G.degree(u) * G.degree(v)
    return _apply_prediction(G, predict, ebunch)