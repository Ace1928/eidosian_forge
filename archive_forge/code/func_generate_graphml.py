import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def generate_graphml(G, encoding='utf-8', prettyprint=True, named_key_ids=False, edge_id_from_attribute=None):
    """Generate GraphML lines for G

    Parameters
    ----------
    G : graph
       A networkx graph
    encoding : string (optional)
       Encoding for text data.
    prettyprint : bool (optional)
       If True use line breaks and indenting in output XML.
    named_key_ids : bool (optional)
       If True use attr.name as value for key elements' id attribute.
    edge_id_from_attribute : dict key (optional)
        If provided, the graphml edge id is set by looking up the corresponding
        edge data attribute keyed by this parameter. If `None` or the key does not exist in edge data,
        the edge id is set by the edge key if `G` is a MultiGraph, else the edge id is left unset.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> linefeed = chr(10)  # linefeed = 

    >>> s = linefeed.join(nx.generate_graphml(G))
    >>> for line in nx.generate_graphml(G):  # doctest: +SKIP
    ...     print(line)

    Notes
    -----
    This implementation does not support mixed graphs (directed and unidirected
    edges together) hyperedges, nested graphs, or ports.
    """
    writer = GraphMLWriter(encoding=encoding, prettyprint=prettyprint, named_key_ids=named_key_ids, edge_id_from_attribute=edge_id_from_attribute)
    writer.add_graph_element(G)
    yield from str(writer).splitlines()