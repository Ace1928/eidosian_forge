import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
@nx._dispatch
def simrank_similarity(G, source=None, target=None, importance_factor=0.9, max_iterations=1000, tolerance=0.0001):
    """Returns the SimRank similarity of nodes in the graph ``G``.

    SimRank is a similarity metric that says "two objects are considered
    to be similar if they are referenced by similar objects." [1]_.

    The pseudo-code definition from the paper is::

        def simrank(G, u, v):
            in_neighbors_u = G.predecessors(u)
            in_neighbors_v = G.predecessors(v)
            scale = C / (len(in_neighbors_u) * len(in_neighbors_v))
            return scale * sum(simrank(G, w, x)
                               for w, x in product(in_neighbors_u,
                                                   in_neighbors_v))

    where ``G`` is the graph, ``u`` is the source, ``v`` is the target,
    and ``C`` is a float decay or importance factor between 0 and 1.

    The SimRank algorithm for determining node similarity is defined in
    [2]_.

    Parameters
    ----------
    G : NetworkX graph
        A NetworkX graph

    source : node
        If this is specified, the returned dictionary maps each node
        ``v`` in the graph to the similarity between ``source`` and
        ``v``.

    target : node
        If both ``source`` and ``target`` are specified, the similarity
        value between ``source`` and ``target`` is returned. If
        ``target`` is specified but ``source`` is not, this argument is
        ignored.

    importance_factor : float
        The relative importance of indirect neighbors with respect to
        direct neighbors.

    max_iterations : integer
        Maximum number of iterations.

    tolerance : float
        Error tolerance used to check convergence. When an iteration of
        the algorithm finds that no similarity value changes more than
        this amount, the algorithm halts.

    Returns
    -------
    similarity : dictionary or float
        If ``source`` and ``target`` are both ``None``, this returns a
        dictionary of dictionaries, where keys are node pairs and value
        are similarity of the pair of nodes.

        If ``source`` is not ``None`` but ``target`` is, this returns a
        dictionary mapping node to the similarity of ``source`` and that
        node.

        If neither ``source`` nor ``target`` is ``None``, this returns
        the similarity value for the given pair of nodes.

    Examples
    --------
    >>> G = nx.cycle_graph(2)
    >>> nx.simrank_similarity(G)
    {0: {0: 1.0, 1: 0.0}, 1: {0: 0.0, 1: 1.0}}
    >>> nx.simrank_similarity(G, source=0)
    {0: 1.0, 1: 0.0}
    >>> nx.simrank_similarity(G, source=0, target=0)
    1.0

    The result of this function can be converted to a numpy array
    representing the SimRank matrix by using the node order of the
    graph to determine which row and column represent each node.
    Other ordering of nodes is also possible.

    >>> import numpy as np
    >>> sim = nx.simrank_similarity(G)
    >>> np.array([[sim[u][v] for v in G] for u in G])
    array([[1., 0.],
           [0., 1.]])
    >>> sim_1d = nx.simrank_similarity(G, source=0)
    >>> np.array([sim[0][v] for v in G])
    array([1., 0.])

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/SimRank
    .. [2] G. Jeh and J. Widom.
           "SimRank: a measure of structural-context similarity",
           In KDD'02: Proceedings of the Eighth ACM SIGKDD
           International Conference on Knowledge Discovery and Data Mining,
           pp. 538--543. ACM Press, 2002.
    """
    import numpy as np
    nodelist = list(G)
    s_indx = None if source is None else nodelist.index(source)
    t_indx = None if target is None else nodelist.index(target)
    x = _simrank_similarity_numpy(G, s_indx, t_indx, importance_factor, max_iterations, tolerance)
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return dict(zip(G, x))
        return {u: dict(zip(G, row)) for u, row in zip(G, x)}
    return x