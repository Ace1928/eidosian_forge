import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
def relabel_gexf_graph(G):
    """Relabel graph using "label" node keyword for node label.

    Parameters
    ----------
    G : graph
       A NetworkX graph read from GEXF data

    Returns
    -------
    H : graph
      A NetworkX graph with relabeled nodes

    Raises
    ------
    NetworkXError
        If node labels are missing or not unique while relabel=True.

    Notes
    -----
    This function relabels the nodes in a NetworkX graph with the
    "label" attribute.  It also handles relabeling the specific GEXF
    node attributes "parents", and "pid".
    """
    try:
        mapping = [(u, G.nodes[u]['label']) for u in G]
    except KeyError as err:
        raise nx.NetworkXError('Failed to relabel nodes: missing node labels found. Use relabel=False.') from err
    x, y = zip(*mapping)
    if len(set(y)) != len(G):
        raise nx.NetworkXError('Failed to relabel nodes: duplicate node labels found. Use relabel=False.')
    mapping = dict(mapping)
    H = nx.relabel_nodes(G, mapping)
    for n in G:
        m = mapping[n]
        H.nodes[m]['id'] = n
        H.nodes[m].pop('label')
        if 'pid' in H.nodes[m]:
            H.nodes[m]['pid'] = mapping[G.nodes[n]['pid']]
        if 'parents' in H.nodes[m]:
            H.nodes[m]['parents'] = [mapping[p] for p in G.nodes[n]['parents']]
    return H