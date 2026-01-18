from collections import defaultdict
import networkx as nx
@nx._dispatch
def check_planarity_recursive(G, counterexample=False):
    """Recursive version of :meth:`check_planarity`."""
    planarity_state = LRPlanarity(G)
    embedding = planarity_state.lr_planarity_recursive()
    if embedding is None:
        if counterexample:
            return (False, get_counterexample_recursive(G))
        else:
            return (False, None)
    else:
        return (True, embedding)