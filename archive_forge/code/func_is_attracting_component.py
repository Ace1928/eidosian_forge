import networkx as nx
from networkx.utils.decorators import not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def is_attracting_component(G):
    """Returns True if `G` consists of a single attracting component.

    Parameters
    ----------
    G : DiGraph, MultiDiGraph
        The graph to be analyzed.

    Returns
    -------
    attracting : bool
        True if `G` has a single attracting component. Otherwise, False.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is undirected.

    See Also
    --------
    attracting_components
    number_attracting_components

    """
    ac = list(attracting_components(G))
    if len(ac) == 1:
        return len(ac[0]) == len(G)
    return False