import sys
import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def forest_str(graph, with_labels=True, sources=None, write=None, ascii_only=False):
    """Creates a nice utf8 representation of a forest

    This function has been superseded by
    :func:`nx.readwrite.text.generate_network_text`, which should be used
    instead.

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent (must be a tree, forest, or the empty graph)

    with_labels : bool
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. Defaults to True.

    sources : List
        Mainly relevant for undirected forests, specifies which nodes to list
        first. If unspecified the root nodes of each tree will be used for
        directed forests; for undirected forests this defaults to the nodes
        with the smallest degree.

    write : callable
        Function to use to write to, if None new lines are appended to
        a list and returned. If set to the `print` function, lines will
        be written to stdout as they are generated. If specified,
        this function will return None. Defaults to None.

    ascii_only : Boolean
        If True only ASCII characters are used to construct the visualization

    Returns
    -------
    str | None :
        utf8 representation of the tree / forest

    Examples
    --------
    >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
    >>> print(nx.forest_str(graph))
    ╙── 0
        ├─╼ 1
        │   ├─╼ 3
        │   │   ├─╼ 7
        │   │   └─╼ 8
        │   └─╼ 4
        │       ├─╼ 9
        │       └─╼ 10
        └─╼ 2
            ├─╼ 5
            │   ├─╼ 11
            │   └─╼ 12
            └─╼ 6
                ├─╼ 13
                └─╼ 14


    >>> graph = nx.balanced_tree(r=1, h=2, create_using=nx.Graph)
    >>> print(nx.forest_str(graph))
    ╙── 0
        └── 1
            └── 2

    >>> print(nx.forest_str(graph, ascii_only=True))
    +-- 0
        L-- 1
            L-- 2
    """
    msg = '\nforest_str is deprecated as of version 3.1 and will be removed in version 3.3. Use generate_network_text or write_network_text instead.\n'
    warnings.warn(msg, DeprecationWarning)
    if len(graph.nodes) > 0:
        if not nx.is_forest(graph):
            raise nx.NetworkXNotImplemented('input must be a forest or the empty graph')
    printbuf = []
    if write is None:
        _write = printbuf.append
    else:
        _write = write
    write_network_text(graph, _write, with_labels=with_labels, sources=sources, ascii_only=ascii_only, end='')
    if write is None:
        return '\n'.join(printbuf)