import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
@py_random_state(1)
@nx._dispatch(graphs=None)
def random_tree(n, seed=None, create_using=None):
    """Returns a uniformly random tree on `n` nodes.

    .. deprecated:: 3.2

       ``random_tree`` is deprecated and will be removed in NX v3.4
       Use ``random_labeled_tree`` instead.

    Parameters
    ----------
    n : int
        A positive integer representing the number of nodes in the tree.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX graph
        A tree, given as an undirected graph, whose nodes are numbers in
        the set {0, …, *n* - 1}.

    Raises
    ------
    NetworkXPointlessConcept
        If `n` is zero (because the null graph is not a tree).

    Notes
    -----
    The current implementation of this function generates a uniformly
    random Prüfer sequence then converts that to a tree via the
    :func:`~networkx.from_prufer_sequence` function. Since there is a
    bijection between Prüfer sequences of length *n* - 2 and trees on
    *n* nodes, the tree is chosen uniformly at random from the set of
    all trees on *n* nodes.

    Examples
    --------
    >>> tree = nx.random_tree(n=10, seed=0)
    >>> nx.write_network_text(tree, sources=[0])
    ╙── 0
        ├── 3
        └── 4
            ├── 6
            │   ├── 1
            │   ├── 2
            │   └── 7
            │       └── 8
            │           └── 5
            └── 9

    >>> tree = nx.random_tree(n=10, seed=0, create_using=nx.DiGraph)
    >>> nx.write_network_text(tree)
    ╙── 0
        ├─╼ 3
        └─╼ 4
            ├─╼ 6
            │   ├─╼ 1
            │   ├─╼ 2
            │   └─╼ 7
            │       └─╼ 8
            │           └─╼ 5
            └─╼ 9
    """
    warnings.warn('\n\nrandom_tree is deprecated and will be removed in NX v3.4\nUse random_labeled_tree instead.', DeprecationWarning, stacklevel=2)
    if n == 0:
        raise nx.NetworkXPointlessConcept('the null graph is not a tree')
    if n == 1:
        utree = nx.empty_graph(1, create_using)
    else:
        sequence = [seed.choice(range(n)) for i in range(n - 2)]
        utree = nx.from_prufer_sequence(sequence)
    if create_using is None:
        tree = utree
    else:
        tree = nx.empty_graph(0, create_using)
        if tree.is_directed():
            edges = nx.dfs_edges(utree, source=0)
        else:
            edges = utree.edges
        tree.add_nodes_from(utree.nodes)
        tree.add_edges_from(edges)
    return tree