def back_edge(self, e):
    """
        This is invoked on the back edges in the graph.
        For an undirected graph there is some ambiguity between tree edges
        and back edges since the edge :math:`(u, v)` and :math:`(v, u)` are the
        same edge, but both the `tree_edge()` and `back_edge()` functions will be
        invoked. One way to resolve this ambiguity is to record the tree edges,
        and then disregard the back-edges that are already marked as tree edges.
        An easy way to record tree edges is to record predecessors at the
        `tree_edge` event point.
        """
    return