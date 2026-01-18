import numbers
from collections import Counter
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import discrete_sequence, py_random_state, weighted_choice
@py_random_state(3)
@nx._dispatch(graphs=None)
def gn_graph(n, kernel=None, create_using=None, seed=None):
    """Returns the growing network (GN) digraph with `n` nodes.

    The GN graph is built by adding nodes one at a time with a link to one
    previously added node.  The target node for the link is chosen with
    probability based on degree.  The default attachment kernel is a linear
    function of the degree of a node.

    The graph is always a (directed) tree.

    Parameters
    ----------
    n : int
        The number of nodes for the generated graph.
    kernel : function
        The attachment kernel.
    create_using : NetworkX graph constructor, optional (default DiGraph)
        Graph type to create. If graph instance, then cleared before populated.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Examples
    --------
    To create the undirected GN graph, use the :meth:`~DiGraph.to_directed`
    method::

    >>> D = nx.gn_graph(10)  # the GN graph
    >>> G = D.to_undirected()  # the undirected version

    To specify an attachment kernel, use the `kernel` keyword argument::

    >>> D = nx.gn_graph(10, kernel=lambda x: x ** 1.5)  # A_k = k^1.5

    References
    ----------
    .. [1] P. L. Krapivsky and S. Redner,
           Organization of Growing Random Networks,
           Phys. Rev. E, 63, 066123, 2001.
    """
    G = empty_graph(1, create_using, default=nx.DiGraph)
    if not G.is_directed():
        raise nx.NetworkXError('create_using must indicate a Directed Graph')
    if kernel is None:

        def kernel(x):
            return x
    if n == 1:
        return G
    G.add_edge(1, 0)
    ds = [1, 1]
    for source in range(2, n):
        dist = [kernel(d) for d in ds]
        target = discrete_sequence(1, distribution=dist, seed=seed)[0]
        G.add_edge(source, target)
        ds.append(1)
        ds[target] += 1
    return G