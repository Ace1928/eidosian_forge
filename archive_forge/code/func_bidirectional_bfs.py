import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
def bidirectional_bfs():
    """Bidirectional breadth-first search for an augmenting path."""
    pred = {s: None}
    q_s = [s]
    succ = {t: None}
    q_t = [t]
    while True:
        q = []
        if len(q_s) <= len(q_t):
            for u in q_s:
                for v, attr in R_succ[u].items():
                    if v not in pred and attr['flow'] < attr['capacity']:
                        pred[v] = u
                        if v in succ:
                            return (v, pred, succ)
                        q.append(v)
            if not q:
                return (None, None, None)
            q_s = q
        else:
            for u in q_t:
                for v, attr in R_pred[u].items():
                    if v not in succ and attr['flow'] < attr['capacity']:
                        succ[v] = u
                        if v in pred:
                            return (v, pred, succ)
                        q.append(v)
            if not q:
                return (None, None, None)
            q_t = q