import networkx as nx
@nx._dispatch(edge_attrs={'weight': 0})
def cost_of_flow(G, flowDict, weight='weight'):
    """Compute the cost of the flow given by flowDict on graph G.

    Note that this function does not check for the validity of the
    flow flowDict. This function will fail if the graph G and the
    flow don't have the same edge set.

    Parameters
    ----------
    G : NetworkX graph
        DiGraph on which a minimum cost flow satisfying all demands is
        to be found.

    weight : string
        Edges of the graph G are expected to have an attribute weight
        that indicates the cost incurred by sending one unit of flow on
        that edge. If not present, the weight is considered to be 0.
        Default value: 'weight'.

    flowDict : dictionary
        Dictionary of dictionaries keyed by nodes such that
        flowDict[u][v] is the flow edge (u, v).

    Returns
    -------
    cost : Integer, float
        The total cost of the flow. This is given by the sum over all
        edges of the product of the edge's flow and the edge's weight.

    See also
    --------
    max_flow_min_cost, min_cost_flow, min_cost_flow_cost, network_simplex

    Notes
    -----
    This algorithm is not guaranteed to work if edge weights or demands
    are floating point numbers (overflows and roundoff errors can
    cause problems). As a workaround you can use integer numbers by
    multiplying the relevant edge attributes by a convenient
    constant factor (eg 100).
    """
    return sum((flowDict[u][v] * d.get(weight, 0) for u, v, d in G.edges(data=True)))