import itertools
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
import networkx as nx
from networkx.algorithms.approximation import local_node_connectivity
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
def adjacency(self):
    """Returns an iterator of (node, adjacency set) tuples for all nodes
           in the dense graph.

        This is the fastest way to look at every edge.
        For directed graphs, only outgoing adjacencies are included.

        Returns
        -------
        adj_iter : iterator
           An iterator of (node, adjacency set) for all nodes in
           the graph.

        """
    for n in self._adj:
        yield (n, set(self._adj) - set(self._adj[n]) - {n})