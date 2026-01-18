from collections import defaultdict
import networkx as nx
@nx._dispatch
def check_planarity(G, counterexample=False):
    """Check if a graph is planar and return a counterexample or an embedding.

    A graph is planar iff it can be drawn in a plane without
    any edge intersections.

    Parameters
    ----------
    G : NetworkX graph
    counterexample : bool
        A Kuratowski subgraph (to proof non planarity) is only returned if set
        to true.

    Returns
    -------
    (is_planar, certificate) : (bool, NetworkX graph) tuple
        is_planar is true if the graph is planar.
        If the graph is planar `certificate` is a PlanarEmbedding
        otherwise it is a Kuratowski subgraph.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2)])
    >>> is_planar, P = nx.check_planarity(G)
    >>> print(is_planar)
    True

    When `G` is planar, a `PlanarEmbedding` instance is returned:

    >>> P.get_data()
    {0: [1, 2], 1: [0], 2: [0]}

    Notes
    -----
    A (combinatorial) embedding consists of cyclic orderings of the incident
    edges at each vertex. Given such an embedding there are multiple approaches
    discussed in literature to drawing the graph (subject to various
    constraints, e.g. integer coordinates), see e.g. [2].

    The planarity check algorithm and extraction of the combinatorial embedding
    is based on the Left-Right Planarity Test [1].

    A counterexample is only generated if the corresponding parameter is set,
    because the complexity of the counterexample generation is higher.

    See also
    --------
    is_planar :
        Check for planarity without creating a `PlanarEmbedding` or counterexample.

    References
    ----------
    .. [1] Ulrik Brandes:
        The Left-Right Planarity Test
        2009
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.217.9208
    .. [2] Takao Nishizeki, Md Saidur Rahman:
        Planar graph drawing
        Lecture Notes Series on Computing: Volume 12
        2004
    """
    planarity_state = LRPlanarity(G)
    embedding = planarity_state.lr_planarity()
    if embedding is None:
        if counterexample:
            return (False, get_counterexample(G))
        else:
            return (False, None)
    else:
        return (True, embedding)