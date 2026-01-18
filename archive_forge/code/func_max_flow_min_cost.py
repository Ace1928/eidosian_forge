import networkx as nx
@nx._dispatch(edge_attrs={'capacity': float('inf'), 'weight': 0})
def max_flow_min_cost(G, s, t, capacity='capacity', weight='weight'):
    """Returns a maximum (s, t)-flow of minimum cost.

    G is a digraph with edge costs and capacities. There is a source
    node s and a sink node t. This function finds a maximum flow from
    s to t whose total cost is minimized.

    Parameters
    ----------
    G : NetworkX graph
        DiGraph on which a minimum cost flow satisfying all demands is
        to be found.

    s: node label
        Source of the flow.

    t: node label
        Destination of the flow.

    capacity: string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    weight: string
        Edges of the graph G are expected to have an attribute weight
        that indicates the cost incurred by sending one unit of flow on
        that edge. If not present, the weight is considered to be 0.
        Default value: 'weight'.

    Returns
    -------
    flowDict: dictionary
        Dictionary of dictionaries keyed by nodes such that
        flowDict[u][v] is the flow edge (u, v).

    Raises
    ------
    NetworkXError
        This exception is raised if the input graph is not directed or
        not connected.

    NetworkXUnbounded
        This exception is raised if there is an infinite capacity path
        from s to t in G. In this case there is no maximum flow. This
        exception is also raised if the digraph G has a cycle of
        negative cost and infinite capacity. Then, the cost of a flow
        is unbounded below.

    See also
    --------
    cost_of_flow, min_cost_flow, min_cost_flow_cost, network_simplex

    Notes
    -----
    This algorithm is not guaranteed to work if edge weights or demands
    are floating point numbers (overflows and roundoff errors can
    cause problems). As a workaround you can use integer numbers by
    multiplying the relevant edge attributes by a convenient
    constant factor (eg 100).

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from(
    ...     [
    ...         (1, 2, {"capacity": 12, "weight": 4}),
    ...         (1, 3, {"capacity": 20, "weight": 6}),
    ...         (2, 3, {"capacity": 6, "weight": -3}),
    ...         (2, 6, {"capacity": 14, "weight": 1}),
    ...         (3, 4, {"weight": 9}),
    ...         (3, 5, {"capacity": 10, "weight": 5}),
    ...         (4, 2, {"capacity": 19, "weight": 13}),
    ...         (4, 5, {"capacity": 4, "weight": 0}),
    ...         (5, 7, {"capacity": 28, "weight": 2}),
    ...         (6, 5, {"capacity": 11, "weight": 1}),
    ...         (6, 7, {"weight": 8}),
    ...         (7, 4, {"capacity": 6, "weight": 6}),
    ...     ]
    ... )
    >>> mincostFlow = nx.max_flow_min_cost(G, 1, 7)
    >>> mincost = nx.cost_of_flow(G, mincostFlow)
    >>> mincost
    373
    >>> from networkx.algorithms.flow import maximum_flow
    >>> maxFlow = maximum_flow(G, 1, 7)[1]
    >>> nx.cost_of_flow(G, maxFlow) >= mincost
    True
    >>> mincostFlowValue = sum((mincostFlow[u][7] for u in G.predecessors(7))) - sum(
    ...     (mincostFlow[7][v] for v in G.successors(7))
    ... )
    >>> mincostFlowValue == nx.maximum_flow_value(G, 1, 7)
    True

    """
    maxFlow = nx.maximum_flow_value(G, s, t, capacity=capacity)
    H = nx.DiGraph(G)
    H.add_node(s, demand=-maxFlow)
    H.add_node(t, demand=maxFlow)
    return min_cost_flow(H, capacity=capacity, weight=weight)