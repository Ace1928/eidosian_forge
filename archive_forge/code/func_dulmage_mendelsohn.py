from pyomo.common.dependencies import networkx_available
def dulmage_mendelsohn(bg, top_nodes=None, matching=None):
    """
    The Dulmage-Mendelsohn decomposition for bipartite graphs.
    This is the coarse decomposition.
    """
    top, bot = bipartite_sets(bg, top_nodes)
    bot_nodes = [n for n in bg if n not in top]
    if top_nodes is None:
        top_nodes = [n for n in bg if n in top]
    if matching is None:
        matching = maximum_matching(bg, top_nodes=top_nodes)
    t_unmatched = [t for t in top_nodes if t not in matching]
    b_unmatched = [b for b in bot_nodes if b not in matching]
    t_digraph = _get_projected_digraph(bg, matching, top_nodes)
    b_digraph = _get_projected_digraph(bg, matching, bot_nodes)
    t_reachable, t_filter = _get_reachable_from(t_digraph, t_unmatched)
    b_reachable, b_filter = _get_reachable_from(b_digraph, b_unmatched)
    t_matched_with_reachable = [matching[b] for b in b_reachable]
    b_matched_with_reachable = [matching[t] for t in t_reachable]
    _filter = t_filter.union(b_filter)
    _filter.update(t_unmatched)
    _filter.update(t_matched_with_reachable)
    _filter.update(b_unmatched)
    _filter.update(b_matched_with_reachable)
    t_other = [t for t in top_nodes if t not in _filter]
    b_other = [matching[t] for t in t_other]
    return ((t_unmatched, t_reachable, t_matched_with_reachable, t_other), (b_unmatched, b_reachable, b_matched_with_reachable, b_other))