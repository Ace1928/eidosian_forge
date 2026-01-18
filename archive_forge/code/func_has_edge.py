from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.reportviews import DegreeView, EdgeView, NodeView
from networkx.exception import NetworkXError
def has_edge(self, u, v):
    """Returns True if the edge (u, v) is in the graph.

        This is the same as `v in G[u]` without KeyError exceptions.

        Parameters
        ----------
        u, v : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        edge_ind : bool
            True if edge is in the graph, False otherwise.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.has_edge(0, 1)  # using two nodes
        True
        >>> e = (0, 1)
        >>> G.has_edge(*e)  #  e is a 2-tuple (u, v)
        True
        >>> e = (0, 1, {"weight": 7})
        >>> G.has_edge(*e[:2])  # e is a 3-tuple (u, v, data_dictionary)
        True

        The following syntax are equivalent:

        >>> G.has_edge(0, 1)
        True
        >>> 1 in G[0]  # though this gives KeyError if 0 not in G
        True

        """
    try:
        return v in self._adj[u]
    except KeyError:
        return False