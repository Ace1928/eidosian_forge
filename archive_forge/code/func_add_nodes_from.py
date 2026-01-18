from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.reportviews import DegreeView, EdgeView, NodeView
from networkx.exception import NetworkXError
def add_nodes_from(self, nodes_for_adding, **attr):
    """Add multiple nodes.

        Parameters
        ----------
        nodes_for_adding : iterable container
            A container of nodes (list, dict, set, etc.).
            OR
            A container of (node, attribute dict) tuples.
            Node attributes are updated using the attribute dict.
        attr : keyword arguments, optional (default= no attributes)
            Update attributes for all nodes in nodes.
            Node attributes specified in nodes as a tuple take
            precedence over attributes specified via keyword arguments.

        See Also
        --------
        add_node

        Notes
        -----
        When adding nodes from an iterator over the graph you are changing,
        a `RuntimeError` can be raised with message:
        `RuntimeError: dictionary changed size during iteration`. This
        happens when the graph's underlying dictionary is modified during
        iteration. To avoid this error, evaluate the iterator into a separate
        object, e.g. by using `list(iterator_of_nodes)`, and pass this
        object to `G.add_nodes_from`.

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_nodes_from("Hello")
        >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> G.add_nodes_from(K3)
        >>> sorted(G.nodes(), key=str)
        [0, 1, 2, 'H', 'e', 'l', 'o']

        Use keywords to update specific node attributes for every node.

        >>> G.add_nodes_from([1, 2], size=10)
        >>> G.add_nodes_from([3, 4], weight=0.4)

        Use (node, attrdict) tuples to update attributes for specific nodes.

        >>> G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])
        >>> G.nodes[1]["size"]
        11
        >>> H = nx.Graph()
        >>> H.add_nodes_from(G.nodes(data=True))
        >>> H.nodes[1]["size"]
        11

        Evaluate an iterator over a graph if using it to modify the same graph

        >>> G = nx.Graph([(0, 1), (1, 2), (3, 4)])
        >>> # wrong way - will raise RuntimeError
        >>> # G.add_nodes_from(n + 1 for n in G.nodes)
        >>> # correct way
        >>> G.add_nodes_from(list(n + 1 for n in G.nodes))
        """
    for n in nodes_for_adding:
        try:
            newnode = n not in self._node
            newdict = attr
        except TypeError:
            n, ndict = n
            newnode = n not in self._node
            newdict = attr.copy()
            newdict.update(ndict)
        if newnode:
            if n is None:
                raise ValueError('None cannot be a node')
            self._adj[n] = self.adjlist_inner_dict_factory()
            self._node[n] = self.node_attr_dict_factory()
        self._node[n].update(newdict)