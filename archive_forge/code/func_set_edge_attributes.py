from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def set_edge_attributes(G, values, name=None):
    """Sets edge attributes from a given value or dictionary of values.

    .. Warning:: The call order of arguments `values` and `name`
        switched between v1.x & v2.x.

    Parameters
    ----------
    G : NetworkX Graph

    values : scalar value, dict-like
        What the edge attribute should be set to.  If `values` is
        not a dictionary, then it is treated as a single attribute value
        that is then applied to every edge in `G`.  This means that if
        you provide a mutable object, like a list, updates to that object
        will be reflected in the edge attribute for each edge.  The attribute
        name will be `name`.

        If `values` is a dict or a dict of dict, it should be keyed
        by edge tuple to either an attribute value or a dict of attribute
        key/value pairs used to update the edge's attributes.
        For multigraphs, the edge tuples must be of the form ``(u, v, key)``,
        where `u` and `v` are nodes and `key` is the edge key.
        For non-multigraphs, the keys must be tuples of the form ``(u, v)``.

    name : string (optional, default=None)
        Name of the edge attribute to set if values is a scalar.

    Examples
    --------
    After computing some property of the edges of a graph, you may want
    to assign a edge attribute to store the value of that property for
    each edge::

        >>> G = nx.path_graph(3)
        >>> bb = nx.edge_betweenness_centrality(G, normalized=False)
        >>> nx.set_edge_attributes(G, bb, "betweenness")
        >>> G.edges[1, 2]["betweenness"]
        2.0

    If you provide a list as the second argument, updates to the list
    will be reflected in the edge attribute for each edge::

        >>> labels = []
        >>> nx.set_edge_attributes(G, labels, "labels")
        >>> labels.append("foo")
        >>> G.edges[0, 1]["labels"]
        ['foo']
        >>> G.edges[1, 2]["labels"]
        ['foo']

    If you provide a dictionary of dictionaries as the second argument,
    the entire dictionary will be used to update edge attributes::

        >>> G = nx.path_graph(3)
        >>> attrs = {(0, 1): {"attr1": 20, "attr2": "nothing"}, (1, 2): {"attr2": 3}}
        >>> nx.set_edge_attributes(G, attrs)
        >>> G[0][1]["attr1"]
        20
        >>> G[0][1]["attr2"]
        'nothing'
        >>> G[1][2]["attr2"]
        3

    The attributes of one Graph can be used to set those of another.

        >>> H = nx.path_graph(3)
        >>> nx.set_edge_attributes(H, G.edges)

    Note that if the dict contains edges that are not in `G`, they are
    silently ignored::

        >>> G = nx.Graph([(0, 1)])
        >>> nx.set_edge_attributes(G, {(1, 2): {"weight": 2.0}})
        >>> (1, 2) in G.edges()
        False

    For multigraphs, the `values` dict is expected to be keyed by 3-tuples
    including the edge key::

        >>> MG = nx.MultiGraph()
        >>> edges = [(0, 1), (0, 1)]
        >>> MG.add_edges_from(edges)  # Returns list of edge keys
        [0, 1]
        >>> attributes = {(0, 1, 0): {"cost": 21}, (0, 1, 1): {"cost": 7}}
        >>> nx.set_edge_attributes(MG, attributes)
        >>> MG[0][1][0]["cost"]
        21
        >>> MG[0][1][1]["cost"]
        7

    If MultiGraph attributes are desired for a Graph, you must convert the 3-tuple
    multiedge to a 2-tuple edge and the last multiedge's attribute value will
    overwrite the previous values. Continuing from the previous case we get::

        >>> H = nx.path_graph([0, 1, 2])
        >>> nx.set_edge_attributes(H, {(u, v): ed for u, v, ed in MG.edges.data()})
        >>> nx.get_edge_attributes(H, "cost")
        {(0, 1): 7}

    """
    if name is not None:
        try:
            if G.is_multigraph():
                for (u, v, key), value in values.items():
                    try:
                        G[u][v][key][name] = value
                    except KeyError:
                        pass
            else:
                for (u, v), value in values.items():
                    try:
                        G[u][v][name] = value
                    except KeyError:
                        pass
        except AttributeError:
            for u, v, data in G.edges(data=True):
                data[name] = values
    elif G.is_multigraph():
        for (u, v, key), d in values.items():
            try:
                G[u][v][key].update(d)
            except KeyError:
                pass
    else:
        for (u, v), d in values.items():
            try:
                G[u][v].update(d)
            except KeyError:
                pass