import networkx as nx
from networkx.utils import open_file
@nx._dispatch(graphs=None)
def parse_multiline_adjlist(lines, comments='#', delimiter=None, create_using=None, nodetype=None, edgetype=None):
    """Parse lines of a multiline adjacency list representation of a graph.

    Parameters
    ----------
    lines : list or iterator of strings
        Input data in multiline adjlist format

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    nodetype : Python type, optional
       Convert nodes to this type.

    edgetype : Python type, optional
       Convert edges to this type.

    comments : string, optional
       Marker for comment lines

    delimiter : string, optional
       Separator for node labels.  The default is whitespace.

    Returns
    -------
    G: NetworkX graph
        The graph corresponding to the lines in multiline adjacency list format.

    Examples
    --------
    >>> lines = [
    ...     "1 2",
    ...     "2 {'weight':3, 'name': 'Frodo'}",
    ...     "3 {}",
    ...     "2 1",
    ...     "5 {'weight':6, 'name': 'Saruman'}",
    ... ]
    >>> G = nx.parse_multiline_adjlist(iter(lines), nodetype=int)
    >>> list(G)
    [1, 2, 3, 5]

    """
    from ast import literal_eval
    G = nx.empty_graph(0, create_using)
    for line in lines:
        p = line.find(comments)
        if p >= 0:
            line = line[:p]
        if not line:
            continue
        try:
            u, deg = line.strip().split(delimiter)
            deg = int(deg)
        except BaseException as err:
            raise TypeError(f'Failed to read node and degree on line ({line})') from err
        if nodetype is not None:
            try:
                u = nodetype(u)
            except BaseException as err:
                raise TypeError(f'Failed to convert node ({u}) to type {nodetype}') from err
        G.add_node(u)
        for i in range(deg):
            while True:
                try:
                    line = next(lines)
                except StopIteration as err:
                    msg = f'Failed to find neighbor for node ({u})'
                    raise TypeError(msg) from err
                p = line.find(comments)
                if p >= 0:
                    line = line[:p]
                if line:
                    break
            vlist = line.strip().split(delimiter)
            numb = len(vlist)
            if numb < 1:
                continue
            v = vlist.pop(0)
            data = ''.join(vlist)
            if nodetype is not None:
                try:
                    v = nodetype(v)
                except BaseException as err:
                    raise TypeError(f'Failed to convert node ({v}) to type {nodetype}') from err
            if edgetype is not None:
                try:
                    edgedata = {'weight': edgetype(data)}
                except BaseException as err:
                    raise TypeError(f'Failed to convert edge data ({data}) to type {edgetype}') from err
            else:
                try:
                    edgedata = literal_eval(data)
                except:
                    edgedata = {}
            G.add_edge(u, v, **edgedata)
    return G