import networkx as nx
@nx._dispatch(graphs=None)
def nonisomorphic_trees(order, create='graph'):
    """Returns a list of nonisomorphic trees

    Parameters
    ----------
    order : int
      order of the desired tree(s)

    create : graph or matrix (default="Graph)
      If graph is selected a list of trees will be returned,
      if matrix is selected a list of adjacency matrix will
      be returned

    Returns
    -------
    G : List of NetworkX Graphs

    M : List of Adjacency matrices

    References
    ----------

    """
    if order < 2:
        raise ValueError
    layout = list(range(order // 2 + 1)) + list(range(1, (order + 1) // 2))
    while layout is not None:
        layout = _next_tree(layout)
        if layout is not None:
            if create == 'graph':
                yield _layout_to_graph(layout)
            elif create == 'matrix':
                yield _layout_to_matrix(layout)
            layout = _next_rooted_tree(layout)