from itertools import chain
from math import log
import networkx as nx
from ...utils import BinaryHeap, arbitrary_element, not_implemented_for
@nx._dispatch(node_attrs='demand', edge_attrs={'capacity': float('inf'), 'weight': 0})
def capacity_scaling(G, demand='demand', capacity='capacity', weight='weight', heap=BinaryHeap):
    """Find a minimum cost flow satisfying all demands in digraph G.

    This is a capacity scaling successive shortest augmenting path algorithm.

    G is a digraph with edge costs and capacities and in which nodes
    have demand, i.e., they want to send or receive some amount of
    flow. A negative demand means that the node wants to send flow, a
    positive demand means that the node want to receive flow. A flow on
    the digraph G satisfies all demand if the net flow into each node
    is equal to the demand of that node.

    Parameters
    ----------
    G : NetworkX graph
        DiGraph or MultiDiGraph on which a minimum cost flow satisfying all
        demands is to be found.

    demand : string
        Nodes of the graph G are expected to have an attribute demand
        that indicates how much flow a node wants to send (negative
        demand) or receive (positive demand). Note that the sum of the
        demands should be 0 otherwise the problem in not feasible. If
        this attribute is not present, a node is considered to have 0
        demand. Default value: 'demand'.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    weight : string
        Edges of the graph G are expected to have an attribute weight
        that indicates the cost incurred by sending one unit of flow on
        that edge. If not present, the weight is considered to be 0.
        Default value: 'weight'.

    heap : class
        Type of heap to be used in the algorithm. It should be a subclass of
        :class:`MinHeap` or implement a compatible interface.

        If a stock heap implementation is to be used, :class:`BinaryHeap` is
        recommended over :class:`PairingHeap` for Python implementations without
        optimized attribute accesses (e.g., CPython) despite a slower
        asymptotic running time. For Python implementations with optimized
        attribute accesses (e.g., PyPy), :class:`PairingHeap` provides better
        performance. Default value: :class:`BinaryHeap`.

    Returns
    -------
    flowCost : integer
        Cost of a minimum cost flow satisfying all demands.

    flowDict : dictionary
        If G is a digraph, a dict-of-dicts keyed by nodes such that
        flowDict[u][v] is the flow on edge (u, v).
        If G is a MultiDiGraph, a dict-of-dicts-of-dicts keyed by nodes
        so that flowDict[u][v][key] is the flow on edge (u, v, key).

    Raises
    ------
    NetworkXError
        This exception is raised if the input graph is not directed,
        not connected.

    NetworkXUnfeasible
        This exception is raised in the following situations:

            * The sum of the demands is not zero. Then, there is no
              flow satisfying all demands.
            * There is no flow satisfying all demand.

    NetworkXUnbounded
        This exception is raised if the digraph G has a cycle of
        negative cost and infinite capacity. Then, the cost of a flow
        satisfying all demands is unbounded below.

    Notes
    -----
    This algorithm does not work if edge weights are floating-point numbers.

    See also
    --------
    :meth:`network_simplex`

    Examples
    --------
    A simple example of a min cost flow problem.

    >>> G = nx.DiGraph()
    >>> G.add_node("a", demand=-5)
    >>> G.add_node("d", demand=5)
    >>> G.add_edge("a", "b", weight=3, capacity=4)
    >>> G.add_edge("a", "c", weight=6, capacity=10)
    >>> G.add_edge("b", "d", weight=1, capacity=9)
    >>> G.add_edge("c", "d", weight=2, capacity=5)
    >>> flowCost, flowDict = nx.capacity_scaling(G)
    >>> flowCost
    24
    >>> flowDict
    {'a': {'b': 4, 'c': 1}, 'd': {}, 'b': {'d': 4}, 'c': {'d': 1}}

    It is possible to change the name of the attributes used for the
    algorithm.

    >>> G = nx.DiGraph()
    >>> G.add_node("p", spam=-4)
    >>> G.add_node("q", spam=2)
    >>> G.add_node("a", spam=-2)
    >>> G.add_node("d", spam=-1)
    >>> G.add_node("t", spam=2)
    >>> G.add_node("w", spam=3)
    >>> G.add_edge("p", "q", cost=7, vacancies=5)
    >>> G.add_edge("p", "a", cost=1, vacancies=4)
    >>> G.add_edge("q", "d", cost=2, vacancies=3)
    >>> G.add_edge("t", "q", cost=1, vacancies=2)
    >>> G.add_edge("a", "t", cost=2, vacancies=4)
    >>> G.add_edge("d", "w", cost=3, vacancies=4)
    >>> G.add_edge("t", "w", cost=4, vacancies=1)
    >>> flowCost, flowDict = nx.capacity_scaling(
    ...     G, demand="spam", capacity="vacancies", weight="cost"
    ... )
    >>> flowCost
    37
    >>> flowDict
    {'p': {'q': 2, 'a': 2}, 'q': {'d': 1}, 'a': {'t': 4}, 'd': {'w': 2}, 't': {'q': 1, 'w': 1}, 'w': {}}
    """
    R = _build_residual_network(G, demand, capacity, weight)
    inf = float('inf')
    flow_cost = sum((0 if e.get(capacity, inf) <= 0 or e.get(weight, 0) >= 0 else e[capacity] * e[weight] for u, v, e in nx.selfloop_edges(G, data=True)))
    wmax = max(chain([-inf], (e['capacity'] for u, v, e in R.edges(data=True))))
    if wmax == -inf:
        return (flow_cost, _build_flow_dict(G, R, capacity, weight))
    R_nodes = R.nodes
    R_succ = R.succ
    delta = 2 ** int(log(wmax, 2))
    while delta >= 1:
        for u in R:
            p_u = R_nodes[u]['potential']
            for v, es in R_succ[u].items():
                for k, e in es.items():
                    flow = e['capacity'] - e['flow']
                    if e['weight'] - p_u + R_nodes[v]['potential'] < 0:
                        flow = e['capacity'] - e['flow']
                        if flow >= delta:
                            e['flow'] += flow
                            R_succ[v][u][k[0], not k[1]]['flow'] -= flow
                            R_nodes[u]['excess'] -= flow
                            R_nodes[v]['excess'] += flow
        S = set()
        T = set()
        S_add = S.add
        S_remove = S.remove
        T_add = T.add
        T_remove = T.remove
        for u in R:
            excess = R_nodes[u]['excess']
            if excess >= delta:
                S_add(u)
            elif excess <= -delta:
                T_add(u)
        while S and T:
            s = arbitrary_element(S)
            t = None
            d = {}
            pred = {s: None}
            h = heap()
            h_insert = h.insert
            h_get = h.get
            h_insert(s, 0)
            while h:
                u, d_u = h.pop()
                d[u] = d_u
                if u in T:
                    t = u
                    break
                p_u = R_nodes[u]['potential']
                for v, es in R_succ[u].items():
                    if v in d:
                        continue
                    wmin = inf
                    for k, e in es.items():
                        if e['capacity'] - e['flow'] >= delta:
                            w = e['weight']
                            if w < wmin:
                                wmin = w
                                kmin = k
                                emin = e
                    if wmin == inf:
                        continue
                    d_v = d_u + wmin - p_u + R_nodes[v]['potential']
                    if h_insert(v, d_v):
                        pred[v] = (u, kmin, emin)
            if t is not None:
                while u != s:
                    v = u
                    u, k, e = pred[v]
                    e['flow'] += delta
                    R_succ[v][u][k[0], not k[1]]['flow'] -= delta
                R_nodes[s]['excess'] -= delta
                R_nodes[t]['excess'] += delta
                if R_nodes[s]['excess'] < delta:
                    S_remove(s)
                if R_nodes[t]['excess'] > -delta:
                    T_remove(t)
                d_t = d[t]
                for u, d_u in d.items():
                    R_nodes[u]['potential'] -= d_u - d_t
            else:
                S_remove(s)
        delta //= 2
    if any((R.nodes[u]['excess'] != 0 for u in R)):
        raise nx.NetworkXUnfeasible('No flow satisfying all demands.')
    for u in R:
        for v, es in R_succ[u].items():
            for e in es.values():
                flow = e['flow']
                if flow > 0:
                    flow_cost += flow * e['weight']
    return (flow_cost, _build_flow_dict(G, R, capacity, weight))