import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
def edmonds_karp_impl(G, s, t, capacity, residual, cutoff):
    """Implementation of the Edmonds-Karp algorithm."""
    if s not in G:
        raise nx.NetworkXError(f'node {str(s)} not in graph')
    if t not in G:
        raise nx.NetworkXError(f'node {str(t)} not in graph')
    if s == t:
        raise nx.NetworkXError('source and sink are the same node')
    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual
    for u in R:
        for e in R[u].values():
            e['flow'] = 0
    if cutoff is None:
        cutoff = float('inf')
    R.graph['flow_value'] = edmonds_karp_core(R, s, t, cutoff)
    return R