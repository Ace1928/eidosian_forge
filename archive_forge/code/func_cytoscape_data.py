import networkx as nx
def cytoscape_data(G, name='name', ident='id'):
    """Returns data in Cytoscape JSON format (cyjs).

    Parameters
    ----------
    G : NetworkX Graph
        The graph to convert to cytoscape format
    name : string
        A string which is mapped to the 'name' node element in cyjs format.
        Must not have the same value as `ident`.
    ident : string
        A string which is mapped to the 'id' node element in cyjs format.
        Must not have the same value as `name`.

    Returns
    -------
    data: dict
        A dictionary with cyjs formatted data.

    Raises
    ------
    NetworkXError
        If the values for `name` and `ident` are identical.

    See Also
    --------
    cytoscape_graph: convert a dictionary in cyjs format to a graph

    References
    ----------
    .. [1] Cytoscape user's manual:
       http://manual.cytoscape.org/en/stable/index.html

    Examples
    --------
    >>> G = nx.path_graph(2)
    >>> nx.cytoscape_data(G)  # doctest: +SKIP
    {'data': [],
     'directed': False,
     'multigraph': False,
     'elements': {'nodes': [{'data': {'id': '0', 'value': 0, 'name': '0'}},
       {'data': {'id': '1', 'value': 1, 'name': '1'}}],
      'edges': [{'data': {'source': 0, 'target': 1}}]}}
    """
    if name == ident:
        raise nx.NetworkXError('name and ident must be different.')
    jsondata = {'data': list(G.graph.items())}
    jsondata['directed'] = G.is_directed()
    jsondata['multigraph'] = G.is_multigraph()
    jsondata['elements'] = {'nodes': [], 'edges': []}
    nodes = jsondata['elements']['nodes']
    edges = jsondata['elements']['edges']
    for i, j in G.nodes.items():
        n = {'data': j.copy()}
        n['data']['id'] = j.get(ident) or str(i)
        n['data']['value'] = i
        n['data']['name'] = j.get(name) or str(i)
        nodes.append(n)
    if G.is_multigraph():
        for e in G.edges(keys=True):
            n = {'data': G.adj[e[0]][e[1]][e[2]].copy()}
            n['data']['source'] = e[0]
            n['data']['target'] = e[1]
            n['data']['key'] = e[2]
            edges.append(n)
    else:
        for e in G.edges():
            n = {'data': G.adj[e[0]][e[1]].copy()}
            n['data']['source'] = e[0]
            n['data']['target'] = e[1]
            edges.append(n)
    return jsondata