from itertools import combinations
import networkx as nx
from ..utils import arbitrary_element, not_implemented_for
@nx._dispatch
def eulerian_path(G, source=None, keys=False):
    """Return an iterator over the edges of an Eulerian path in `G`.

    Parameters
    ----------
    G : NetworkX Graph
        The graph in which to look for an eulerian path.
    source : node or None (default: None)
        The node at which to start the search. None means search over all
        starting nodes.
    keys : Bool (default: False)
        Indicates whether to yield edge 3-tuples (u, v, edge_key).
        The default yields edge 2-tuples

    Yields
    ------
    Edge tuples along the eulerian path.

    Warning: If `source` provided is not the start node of an Euler path
    will raise error even if an Euler Path exists.
    """
    if not has_eulerian_path(G, source):
        raise nx.NetworkXError('Graph has no Eulerian paths.')
    if G.is_directed():
        G = G.reverse()
        if source is None or nx.is_eulerian(G) is False:
            source = _find_path_start(G)
        if G.is_multigraph():
            for u, v, k in _multigraph_eulerian_circuit(G, source):
                if keys:
                    yield (u, v, k)
                else:
                    yield (u, v)
        else:
            yield from _simplegraph_eulerian_circuit(G, source)
    else:
        G = G.copy()
        if source is None:
            source = _find_path_start(G)
        if G.is_multigraph():
            if keys:
                yield from reversed([(v, u, k) for u, v, k in _multigraph_eulerian_circuit(G, source)])
            else:
                yield from reversed([(v, u) for u, v, k in _multigraph_eulerian_circuit(G, source)])
        else:
            yield from reversed([(v, u) for u, v in _simplegraph_eulerian_circuit(G, source)])