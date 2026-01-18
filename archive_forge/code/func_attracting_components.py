import networkx as nx
from networkx.utils.decorators import not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def attracting_components(G):
    """Generates the attracting components in `G`.

    An attracting component in a directed graph `G` is a strongly connected
    component with the property that a random walker on the graph will never
    leave the component, once it enters the component.

    The nodes in attracting components can also be thought of as recurrent
    nodes.  If a random walker enters the attractor containing the node, then
    the node will be visited infinitely often.

    To obtain induced subgraphs on each component use:
    ``(G.subgraph(c).copy() for c in attracting_components(G))``

    Parameters
    ----------
    G : DiGraph, MultiDiGraph
        The graph to be analyzed.

    Returns
    -------
    attractors : generator of sets
        A generator of sets of nodes, one for each attracting component of G.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is undirected.

    See Also
    --------
    number_attracting_components
    is_attracting_component

    """
    scc = list(nx.strongly_connected_components(G))
    cG = nx.condensation(G, scc)
    for n in cG:
        if cG.out_degree(n) == 0:
            yield scc[n]