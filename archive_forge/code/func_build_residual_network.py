from collections import deque
import networkx as nx
@nx._dispatch(edge_attrs={'capacity': float('inf')})
def build_residual_network(G, capacity):
    """Build a residual network and initialize a zero flow.

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
    stored in :samp:`R.graph['flow_value']`. If :samp:`cutoff` is not
    specified, reachability to :samp:`t` using only edges :samp:`(u, v)` such
    that :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    """
    if G.is_multigraph():
        raise nx.NetworkXError('MultiGraph and MultiDiGraph not supported (yet).')
    R = nx.DiGraph()
    R.add_nodes_from(G)
    inf = float('inf')
    edge_list = [(u, v, attr) for u, v, attr in G.edges(data=True) if u != v and attr.get(capacity, inf) > 0]
    inf = 3 * sum((attr[capacity] for u, v, attr in edge_list if capacity in attr and attr[capacity] != inf)) or 1
    if G.is_directed():
        for u, v, attr in edge_list:
            r = min(attr.get(capacity, inf), inf)
            if not R.has_edge(u, v):
                R.add_edge(u, v, capacity=r)
                R.add_edge(v, u, capacity=0)
            else:
                R[u][v]['capacity'] = r
    else:
        for u, v, attr in edge_list:
            r = min(attr.get(capacity, inf), inf)
            R.add_edge(u, v, capacity=r)
            R.add_edge(v, u, capacity=r)
    R.graph['inf'] = inf
    return R