import networkx as nx
def adjacency_data(G, attrs=_attrs):
    """Returns data in adjacency format that is suitable for JSON serialization
    and use in JavaScript documents.

    Parameters
    ----------
    G : NetworkX graph

    attrs : dict
        A dictionary that contains two keys 'id' and 'key'. The corresponding
        values provide the attribute names for storing NetworkX-internal graph
        data. The values should be unique. Default value:
        :samp:`dict(id='id', key='key')`.

        If some user-defined graph data use these attribute names as data keys,
        they may be silently dropped.

    Returns
    -------
    data : dict
       A dictionary with adjacency formatted data.

    Raises
    ------
    NetworkXError
        If values in attrs are not unique.

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.Graph([(1, 2)])
    >>> data = json_graph.adjacency_data(G)

    To serialize with json

    >>> import json
    >>> s = json.dumps(data)

    Notes
    -----
    Graph, node, and link attributes will be written when using this format
    but attribute keys must be strings if you want to serialize the resulting
    data with JSON.

    The default value of attrs will be changed in a future release of NetworkX.

    See Also
    --------
    adjacency_graph, node_link_data, tree_data
    """
    multigraph = G.is_multigraph()
    id_ = attrs['id']
    key = None if not multigraph else attrs['key']
    if id_ == key:
        raise nx.NetworkXError('Attribute names are not unique.')
    data = {}
    data['directed'] = G.is_directed()
    data['multigraph'] = multigraph
    data['graph'] = list(G.graph.items())
    data['nodes'] = []
    data['adjacency'] = []
    for n, nbrdict in G.adjacency():
        data['nodes'].append({**G.nodes[n], id_: n})
        adj = []
        if multigraph:
            for nbr, keys in nbrdict.items():
                for k, d in keys.items():
                    adj.append({**d, id_: nbr, key: k})
        else:
            for nbr, d in nbrdict.items():
                adj.append({**d, id_: nbr})
        data['adjacency'].append(adj)
    return data