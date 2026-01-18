import itertools as it
from functools import partial
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for
@nx._dispatch
def general_k_edge_subgraphs(G, k):
    """General algorithm to find all maximal k-edge-connected subgraphs in G.

    Returns
    -------
    k_edge_subgraphs : a generator of nx.Graphs that are k-edge-subgraphs
        Each k-edge-subgraph is a maximal set of nodes that defines a subgraph
        of G that is k-edge-connected.

    Notes
    -----
    Implementation of the basic algorithm from _[1].  The basic idea is to find
    a global minimum cut of the graph. If the cut value is at least k, then the
    graph is a k-edge-connected subgraph and can be added to the results.
    Otherwise, the cut is used to split the graph in two and the procedure is
    applied recursively. If the graph is just a single node, then it is also
    added to the results. At the end, each result is either guaranteed to be
    a single node or a subgraph of G that is k-edge-connected.

    This implementation contains optimizations for reducing the number of calls
    to max-flow, but there are other optimizations in _[1] that could be
    implemented.

    References
    ----------
    .. [1] Zhou, Liu, et al. (2012) Finding maximal k-edge-connected subgraphs
        from a large graph.  ACM International Conference on Extending Database
        Technology 2012 480-–491.
        https://openproceedings.org/2012/conf/edbt/ZhouLYLCL12.pdf

    Examples
    --------
    >>> from networkx.utils import pairwise
    >>> paths = [
    ...     (11, 12, 13, 14, 11, 13, 14, 12),  # a 4-clique
    ...     (21, 22, 23, 24, 21, 23, 24, 22),  # another 4-clique
    ...     # connect the cliques with high degree but low connectivity
    ...     (50, 13),
    ...     (12, 50, 22),
    ...     (13, 102, 23),
    ...     (14, 101, 24),
    ... ]
    >>> G = nx.Graph(it.chain(*[pairwise(path) for path in paths]))
    >>> sorted(map(len, k_edge_subgraphs(G, k=3)))
    [1, 1, 1, 4, 4]
    """
    if k < 1:
        raise ValueError('k cannot be less than 1')
    find_ccs = partial(_high_degree_components, k=k)
    if G.number_of_nodes() < k:
        for node in G.nodes():
            yield G.subgraph([node]).copy()
        return
    R0 = {G.subgraph(cc).copy() for cc in find_ccs(G)}
    while R0:
        G1 = R0.pop()
        if G1.number_of_nodes() == 1:
            yield G1
        else:
            cut_edges = nx.minimum_edge_cut(G1)
            cut_value = len(cut_edges)
            if cut_value < k:
                G1.remove_edges_from(cut_edges)
                for cc in find_ccs(G1):
                    R0.add(G1.subgraph(cc).copy())
            else:
                yield G1