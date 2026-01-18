import networkx as nx
from .boykovkolmogorov import boykov_kolmogorov
from .dinitz_alg import dinitz
from .edmondskarp import edmonds_karp
from .preflowpush import preflow_push
from .shortestaugmentingpath import shortest_augmenting_path
from .utils import build_flow_dict
@nx._dispatch(graphs='flowG', edge_attrs={'capacity': float('inf')})
def minimum_cut_value(flowG, _s, _t, capacity='capacity', flow_func=None, **kwargs):
    """Compute the value of a minimum (s, t)-cut.

    Use the max-flow min-cut theorem, i.e., the capacity of a minimum
    capacity cut is equal to the flow value of a maximum flow.

    Parameters
    ----------
    flowG : NetworkX graph
        Edges of the graph are expected to have an attribute called
        'capacity'. If this attribute is not present, the edge is
        considered to have infinite capacity.

    _s : node
        Source node for the flow.

    _t : node
        Sink node for the flow.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes
        in a capacitated graph. The function has to accept at least three
        parameters: a Graph or Digraph, a source node, and a target node.
        And return a residual network that follows NetworkX conventions
        (see Notes). If flow_func is None, the default maximum
        flow function (:meth:`preflow_push`) is used. See below for
        alternative algorithms. The choice of the default function may change
        from version to version and should not be relied on. Default value:
        None.

    kwargs : Any other keyword parameter is passed to the function that
        computes the maximum flow.

    Returns
    -------
    cut_value : integer, float
        Value of the minimum cut.

    Raises
    ------
    NetworkXUnbounded
        If the graph has a path of infinite capacity, all cuts have
        infinite capacity and the function raises a NetworkXError.

    See also
    --------
    :meth:`maximum_flow`
    :meth:`maximum_flow_value`
    :meth:`minimum_cut`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Notes
    -----
    The function used in the flow_func parameter has to return a residual
    network that follows NetworkX conventions:

    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. Reachability to :samp:`t` using
    only edges :samp:`(u, v)` such that
    :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Specific algorithms may store extra data in :samp:`R`.

    The function should supports an optional boolean parameter value_only. When
    True, it can optionally terminate the algorithm as soon as the maximum flow
    value and the minimum cut can be determined.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge("x", "a", capacity=3.0)
    >>> G.add_edge("x", "b", capacity=1.0)
    >>> G.add_edge("a", "c", capacity=3.0)
    >>> G.add_edge("b", "c", capacity=5.0)
    >>> G.add_edge("b", "d", capacity=4.0)
    >>> G.add_edge("d", "e", capacity=2.0)
    >>> G.add_edge("c", "y", capacity=2.0)
    >>> G.add_edge("e", "y", capacity=3.0)

    minimum_cut_value computes only the value of the
    minimum cut:

    >>> cut_value = nx.minimum_cut_value(G, "x", "y")
    >>> cut_value
    3.0

    You can also use alternative algorithms for computing the
    minimum cut by using the flow_func parameter.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> cut_value == nx.minimum_cut_value(
    ...     G, "x", "y", flow_func=shortest_augmenting_path
    ... )
    True

    """
    if flow_func is None:
        if kwargs:
            raise nx.NetworkXError('You have to explicitly set a flow_func if you need to pass parameters via kwargs.')
        flow_func = default_flow_func
    if not callable(flow_func):
        raise nx.NetworkXError('flow_func has to be callable.')
    if kwargs.get('cutoff') is not None and flow_func is preflow_push:
        raise nx.NetworkXError('cutoff should not be specified.')
    R = flow_func(flowG, _s, _t, capacity=capacity, value_only=True, **kwargs)
    return R.graph['flow_value']