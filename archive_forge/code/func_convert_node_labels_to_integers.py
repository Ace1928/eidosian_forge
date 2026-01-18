import networkx as nx
@nx._dispatch(preserve_edge_attrs=True, preserve_node_attrs=True, preserve_graph_attrs=True)
def convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None):
    """Returns a copy of the graph G with the nodes relabeled using
    consecutive integers.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    first_label : int, optional (default=0)
       An integer specifying the starting offset in numbering nodes.
       The new integer labels are numbered first_label, ..., n-1+first_label.

    ordering : string
       "default" : inherit node ordering from G.nodes()
       "sorted"  : inherit node ordering from sorted(G.nodes())
       "increasing degree" : nodes are sorted by increasing degree
       "decreasing degree" : nodes are sorted by decreasing degree

    label_attribute : string, optional (default=None)
       Name of node attribute to store old label.  If None no attribute
       is created.

    Notes
    -----
    Node and edge attribute data are copied to the new (relabeled) graph.

    There is no guarantee that the relabeling of nodes to integers will
    give the same two integers for two (even identical graphs).
    Use the `ordering` argument to try to preserve the order.

    See Also
    --------
    relabel_nodes
    """
    N = G.number_of_nodes() + first_label
    if ordering == 'default':
        mapping = dict(zip(G.nodes(), range(first_label, N)))
    elif ordering == 'sorted':
        nlist = sorted(G.nodes())
        mapping = dict(zip(nlist, range(first_label, N)))
    elif ordering == 'increasing degree':
        dv_pairs = [(d, n) for n, d in G.degree()]
        dv_pairs.sort()
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    elif ordering == 'decreasing degree':
        dv_pairs = [(d, n) for n, d in G.degree()]
        dv_pairs.sort()
        dv_pairs.reverse()
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    else:
        raise nx.NetworkXError(f'Unknown node ordering: {ordering}')
    H = relabel_nodes(G, mapping)
    if label_attribute is not None:
        nx.set_node_attributes(H, {v: k for k, v in mapping.items()}, label_attribute)
    return H