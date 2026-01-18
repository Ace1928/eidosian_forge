import networkx as nx
from networkx.utils.decorators import not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def strongly_connected_components_recursive(G):
    """Generate nodes in strongly connected components of graph.

    .. deprecated:: 3.2

       This function is deprecated and will be removed in a future version of
       NetworkX. Use `strongly_connected_components` instead.

    Recursive version of algorithm.

    Parameters
    ----------
    G : NetworkX Graph
        A directed graph.

    Returns
    -------
    comp : generator of sets
        A generator of sets of nodes, one for each strongly connected
        component of G.

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    Generate a sorted list of strongly connected components, largest first.

    >>> G = nx.cycle_graph(4, create_using=nx.DiGraph())
    >>> nx.add_cycle(G, [10, 11, 12])
    >>> [
    ...     len(c)
    ...     for c in sorted(
    ...         nx.strongly_connected_components_recursive(G), key=len, reverse=True
    ...     )
    ... ]
    [4, 3]

    If you only want the largest component, it's more efficient to
    use max instead of sort.

    >>> largest = max(nx.strongly_connected_components_recursive(G), key=len)

    To create the induced subgraph of the components use:
    >>> S = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

    See Also
    --------
    connected_components

    Notes
    -----
    Uses Tarjan's algorithm[1]_ with Nuutila's modifications[2]_.

    References
    ----------
    .. [1] Depth-first search and linear graph algorithms, R. Tarjan
       SIAM Journal of Computing 1(2):146-160, (1972).

    .. [2] On finding the strongly connected components in a directed graph.
       E. Nuutila and E. Soisalon-Soinen
       Information Processing Letters 49(1): 9-14, (1994)..

    """
    import warnings
    warnings.warn('\n\nstrongly_connected_components_recursive is deprecated and will be\nremoved in the future. Use strongly_connected_components instead.', category=DeprecationWarning, stacklevel=2)
    yield from strongly_connected_components(G)