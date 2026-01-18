import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise
@nx._dispatch
def lexicographical_topological_sort(G, key=None):
    """Generate the nodes in the unique lexicographical topological sort order.

    Generates a unique ordering of nodes by first sorting topologically (for which there are often
    multiple valid orderings) and then additionally by sorting lexicographically.

    A topological sort arranges the nodes of a directed graph so that the
    upstream node of each directed edge precedes the downstream node.
    It is always possible to find a solution for directed graphs that have no cycles.
    There may be more than one valid solution.

    Lexicographical sorting is just sorting alphabetically. It is used here to break ties in the
    topological sort and to determine a single, unique ordering.  This can be useful in comparing
    sort results.

    The lexicographical order can be customized by providing a function to the `key=` parameter.
    The definition of the key function is the same as used in python's built-in `sort()`.
    The function takes a single argument and returns a key to use for sorting purposes.

    Lexicographical sorting can fail if the node names are un-sortable. See the example below.
    The solution is to provide a function to the `key=` argument that returns sortable keys.


    Parameters
    ----------
    G : NetworkX digraph
        A directed acyclic graph (DAG)

    key : function, optional
        A function of one argument that converts a node name to a comparison key.
        It defines and resolves ambiguities in the sort order.  Defaults to the identity function.

    Yields
    ------
    nodes
        Yields the nodes of G in lexicographical topological sort order.

    Raises
    ------
    NetworkXError
        Topological sort is defined for directed graphs only. If the graph `G`
        is undirected, a :exc:`NetworkXError` is raised.

    NetworkXUnfeasible
        If `G` is not a directed acyclic graph (DAG) no topological sort exists
        and a :exc:`NetworkXUnfeasible` exception is raised.  This can also be
        raised if `G` is changed while the returned iterator is being processed

    RuntimeError
        If `G` is changed while the returned iterator is being processed.

    TypeError
        Results from un-sortable node names.
        Consider using `key=` parameter to resolve ambiguities in the sort order.

    Examples
    --------
    >>> DG = nx.DiGraph([(2, 1), (2, 5), (1, 3), (1, 4), (5, 4)])
    >>> list(nx.lexicographical_topological_sort(DG))
    [2, 1, 3, 5, 4]
    >>> list(nx.lexicographical_topological_sort(DG, key=lambda x: -x))
    [2, 5, 1, 4, 3]

    The sort will fail for any graph with integer and string nodes. Comparison of integer to strings
    is not defined in python.  Is 3 greater or less than 'red'?

    >>> DG = nx.DiGraph([(1, 'red'), (3, 'red'), (1, 'green'), (2, 'blue')])
    >>> list(nx.lexicographical_topological_sort(DG))
    Traceback (most recent call last):
    ...
    TypeError: '<' not supported between instances of 'str' and 'int'
    ...

    Incomparable nodes can be resolved using a `key` function. This example function
    allows comparison of integers and strings by returning a tuple where the first
    element is True for `str`, False otherwise. The second element is the node name.
    This groups the strings and integers separately so they can be compared only among themselves.

    >>> key = lambda node: (isinstance(node, str), node)
    >>> list(nx.lexicographical_topological_sort(DG, key=key))
    [1, 2, 3, 'blue', 'green', 'red']

    Notes
    -----
    This algorithm is based on a description and proof in
    "Introduction to Algorithms: A Creative Approach" [1]_ .

    See also
    --------
    topological_sort

    References
    ----------
    .. [1] Manber, U. (1989).
       *Introduction to Algorithms - A Creative Approach.* Addison-Wesley.
    """
    if not G.is_directed():
        msg = 'Topological sort not defined on undirected graphs.'
        raise nx.NetworkXError(msg)
    if key is None:

        def key(node):
            return node
    nodeid_map = {n: i for i, n in enumerate(G)}

    def create_tuple(node):
        return (key(node), nodeid_map[node], node)
    indegree_map = {v: d for v, d in G.in_degree() if d > 0}
    zero_indegree = [create_tuple(v) for v, d in G.in_degree() if d == 0]
    heapq.heapify(zero_indegree)
    while zero_indegree:
        _, _, node = heapq.heappop(zero_indegree)
        if node not in G:
            raise RuntimeError('Graph changed during iteration')
        for _, child in G.edges(node):
            try:
                indegree_map[child] -= 1
            except KeyError as err:
                raise RuntimeError('Graph changed during iteration') from err
            if indegree_map[child] == 0:
                try:
                    heapq.heappush(zero_indegree, create_tuple(child))
                except TypeError as err:
                    raise TypeError(f'{err}\nConsider using `key=` parameter to resolve ambiguities in the sort order.')
                del indegree_map[child]
        yield node
    if indegree_map:
        msg = 'Graph contains a cycle or graph changed during iteration'
        raise nx.NetworkXUnfeasible(msg)