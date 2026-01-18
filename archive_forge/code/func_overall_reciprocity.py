import networkx as nx
from networkx import NetworkXError
from ..utils import not_implemented_for
@not_implemented_for('undirected', 'multigraph')
@nx._dispatch
def overall_reciprocity(G):
    """Compute the reciprocity for the whole graph.

    See the doc of reciprocity for the definition.

    Parameters
    ----------
    G : graph
       A networkx graph

    """
    n_all_edge = G.number_of_edges()
    n_overlap_edge = (n_all_edge - G.to_undirected().number_of_edges()) * 2
    if n_all_edge == 0:
        raise NetworkXError('Not defined for empty graphs')
    return n_overlap_edge / n_all_edge