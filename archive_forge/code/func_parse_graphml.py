import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
@nx._dispatch(graphs=None)
def parse_graphml(graphml_string, node_type=str, edge_key_type=int, force_multigraph=False):
    """Read graph in GraphML format from string.

    Parameters
    ----------
    graphml_string : string
       String containing graphml information
       (e.g., contents of a graphml file).

    node_type: Python type (default: str)
       Convert node ids to this type

    edge_key_type: Python type (default: int)
       Convert graphml edge ids to this type. Multigraphs use id as edge key.
       Non-multigraphs add to edge attribute dict with name "id".

    force_multigraph : bool (default: False)
       If True, return a multigraph with edge keys. If False (the default)
       return a multigraph when multiedges are in the graph.


    Returns
    -------
    graph: NetworkX graph
        If no parallel edges are found a Graph or DiGraph is returned.
        Otherwise a MultiGraph or MultiDiGraph is returned.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> linefeed = chr(10)  # linefeed = 

    >>> s = linefeed.join(nx.generate_graphml(G))
    >>> H = nx.parse_graphml(s)

    Notes
    -----
    Default node and edge attributes are not propagated to each node and edge.
    They can be obtained from `G.graph` and applied to node and edge attributes
    if desired using something like this:

    >>> default_color = G.graph["node_default"]["color"]  # doctest: +SKIP
    >>> for node, data in G.nodes(data=True):  # doctest: +SKIP
    ...     if "color" not in data:
    ...         data["color"] = default_color
    >>> default_color = G.graph["edge_default"]["color"]  # doctest: +SKIP
    >>> for u, v, data in G.edges(data=True):  # doctest: +SKIP
    ...     if "color" not in data:
    ...         data["color"] = default_color

    This implementation does not support mixed graphs (directed and unidirected
    edges together), hypergraphs, nested graphs, or ports.

    For multigraphs the GraphML edge "id" will be used as the edge
    key.  If not specified then they "key" attribute will be used.  If
    there is no "key" attribute a default NetworkX multigraph edge key
    will be provided.

    """
    reader = GraphMLReader(node_type, edge_key_type, force_multigraph)
    glist = list(reader(string=graphml_string))
    if len(glist) == 0:
        header = '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">'
        new_string = graphml_string.replace('<graphml>', header)
        glist = list(reader(string=new_string))
        if len(glist) == 0:
            raise nx.NetworkXError('file not successfully read as graphml')
    return glist[0]