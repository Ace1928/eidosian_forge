import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise
@nx._dispatch
def root_to_leaf_paths(G):
    """Yields root-to-leaf paths in a directed acyclic graph.

    `G` must be a directed acyclic graph. If not, the behavior of this
    function is undefined. A "root" in this graph is a node of in-degree
    zero and a "leaf" a node of out-degree zero.

    When invoked, this function iterates over each path from any root to
    any leaf. A path is a list of nodes.

    """
    roots = (v for v, d in G.in_degree() if d == 0)
    leaves = (v for v, d in G.out_degree() if d == 0)
    all_paths = partial(nx.all_simple_paths, G)
    return chaini(starmap(all_paths, product(roots, leaves)))