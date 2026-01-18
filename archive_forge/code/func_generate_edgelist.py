import networkx as nx
from networkx.utils import not_implemented_for, open_file
@not_implemented_for('directed')
def generate_edgelist(G, delimiter=' ', data=True):
    """Generate a single line of the bipartite graph G in edge list format.

    Parameters
    ----------
    G : NetworkX graph
       The graph is assumed to have node attribute `part` set to 0,1 representing
       the two graph parts

    delimiter : string, optional
       Separator for node labels

    data : bool or list of keys
       If False generate no edge data.  If True use a dictionary
       representation of edge data.  If a list of keys use a list of data
       values corresponding to the keys.

    Returns
    -------
    lines : string
        Lines of data in adjlist format.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> G.add_nodes_from([0, 2], bipartite=0)
    >>> G.add_nodes_from([1, 3], bipartite=1)
    >>> G[1][2]["weight"] = 3
    >>> G[2][3]["capacity"] = 12
    >>> for line in bipartite.generate_edgelist(G, data=False):
    ...     print(line)
    0 1
    2 1
    2 3

    >>> for line in bipartite.generate_edgelist(G):
    ...     print(line)
    0 1 {}
    2 1 {'weight': 3}
    2 3 {'capacity': 12}

    >>> for line in bipartite.generate_edgelist(G, data=["weight"]):
    ...     print(line)
    0 1
    2 1 3
    2 3
    """
    try:
        part0 = [n for n, d in G.nodes.items() if d['bipartite'] == 0]
    except BaseException as err:
        raise AttributeError('Missing node attribute `bipartite`') from err
    if data is True or data is False:
        for n in part0:
            for edge in G.edges(n, data=data):
                yield delimiter.join(map(str, edge))
    else:
        for n in part0:
            for u, v, d in G.edges(n, data=True):
                edge = [u, v]
                try:
                    edge.extend((d[k] for k in data))
                except KeyError:
                    pass
                yield delimiter.join(map(str, edge))