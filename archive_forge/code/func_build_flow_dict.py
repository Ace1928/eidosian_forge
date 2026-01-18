from collections import deque
import networkx as nx
@nx._dispatch(graphs={'G': 0, 'R': 1}, preserve_edge_attrs={'R': {'flow': None}})
def build_flow_dict(G, R):
    """Build a flow dictionary from a residual network."""
    flow_dict = {}
    for u in G:
        flow_dict[u] = {v: 0 for v in G[u]}
        flow_dict[u].update(((v, attr['flow']) for v, attr in R[u].items() if attr['flow'] > 0))
    return flow_dict