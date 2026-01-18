from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import MultiAdjacencyView
from networkx.classes.digraph import DiGraph
from networkx.classes.multigraph import MultiGraph
from networkx.classes.reportviews import (
from networkx.exception import NetworkXError
@cached_property
def out_degree(self):
    """Returns an iterator for (node, out-degree) or out-degree for single node.

        out_degree(self, nbunch=None, weight=None)

        The node out-degree is the number of edges pointing out of the node.
        This function returns the out-degree for a single node or an iterator
        for a bunch of nodes or if nothing is passed as argument.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.

        weight : string or None, optional (default=None)
           The edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.
           The degree is the sum of the edge weights.

        Returns
        -------
        If a single node is requested
        deg : int
            Degree of the node

        OR if multiple nodes are requested
        nd_iter : iterator
            The iterator returns two-tuples of (node, out-degree).

        See Also
        --------
        degree, in_degree

        Examples
        --------
        >>> G = nx.MultiDiGraph()
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.out_degree(0)  # node 0 with degree 1
        1
        >>> list(G.out_degree([0, 1, 2]))
        [(0, 1), (1, 1), (2, 1)]
        >>> G.add_edge(0, 1) # parallel edge
        1
        >>> list(G.out_degree([0, 1, 2])) # counts parallel edges
        [(0, 2), (1, 1), (2, 1)]

        """
    return OutMultiDegreeView(self)