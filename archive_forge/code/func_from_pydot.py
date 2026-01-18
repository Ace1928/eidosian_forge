import warnings
from locale import getpreferredencoding
import networkx as nx
from networkx.utils import open_file
@nx._dispatch(graphs=None)
def from_pydot(P):
    """Returns a NetworkX graph from a Pydot graph.

    Parameters
    ----------
    P : Pydot graph
      A graph created with Pydot

    Returns
    -------
    G : NetworkX multigraph
        A MultiGraph or MultiDiGraph.

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> A = nx.nx_pydot.to_pydot(K5)
    >>> G = nx.nx_pydot.from_pydot(A)  # return MultiGraph

    # make a Graph instead of MultiGraph
    >>> G = nx.Graph(nx.nx_pydot.from_pydot(A))

    """
    msg = 'nx.nx_pydot.from_pydot depends on the pydot package, which has known issues and is not actively maintained.\n\nSee https://github.com/networkx/networkx/issues/5723'
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    if P.get_strict(None):
        multiedges = False
    else:
        multiedges = True
    if P.get_type() == 'graph':
        if multiedges:
            N = nx.MultiGraph()
        else:
            N = nx.Graph()
    elif multiedges:
        N = nx.MultiDiGraph()
    else:
        N = nx.DiGraph()
    name = P.get_name().strip('"')
    if name != '':
        N.name = name
    for p in P.get_node_list():
        n = p.get_name().strip('"')
        if n in ('node', 'graph', 'edge'):
            continue
        N.add_node(n, **p.get_attributes())
    for e in P.get_edge_list():
        u = e.get_source()
        v = e.get_destination()
        attr = e.get_attributes()
        s = []
        d = []
        if isinstance(u, str):
            s.append(u.strip('"'))
        else:
            for unodes in u['nodes']:
                s.append(unodes.strip('"'))
        if isinstance(v, str):
            d.append(v.strip('"'))
        else:
            for vnodes in v['nodes']:
                d.append(vnodes.strip('"'))
        for source_node in s:
            for destination_node in d:
                N.add_edge(source_node, destination_node, **attr)
    pattr = P.get_attributes()
    if pattr:
        N.graph['graph'] = pattr
    try:
        N.graph['node'] = P.get_node_defaults()[0]
    except (IndexError, TypeError):
        pass
    try:
        N.graph['edge'] = P.get_edge_defaults()[0]
    except (IndexError, TypeError):
        pass
    return N