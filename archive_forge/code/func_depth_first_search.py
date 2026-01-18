from collections import deque
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
from networkx.utils import pairwise
def depth_first_search(parents):
    """Build a path using DFS starting from the sink"""
    path = []
    u = t
    flow = INF
    while u != s:
        path.append(u)
        v = parents[u]
        flow = min(flow, R_pred[u][v]['capacity'] - R_pred[u][v]['flow'])
        u = v
    path.append(s)
    if flow > 0:
        for u, v in pairwise(path):
            R_pred[u][v]['flow'] += flow
            R_pred[v][u]['flow'] -= flow
    return flow