import sys
import functools
from .rustworkx import *
import rustworkx.visit
def networkx_converter(graph, keep_attributes: bool=False):
    """Convert a networkx graph object into a rustworkx graph object.

    .. note::

        networkx is **not** a dependency of rustworkx and this function
        is provided as a convenience method for users of both networkx and
        rustworkx. This function will not work unless you install networkx
        independently.

    :param networkx.Graph graph: The networkx graph to convert.
    :param bool keep_attributes: If ``True``, add networkx node attributes to
        the data payload in the nodes of the output rustworkx graph. When set to
        ``True``, the node data payloads in the output rustworkx graph object
        will be dictionaries with the node attributes from the input networkx
        graph where the ``"__networkx_node__"`` key contains the node from the
        input networkx graph.

    :returns: A rustworkx graph, either a :class:`~rustworkx.PyDiGraph` or a
        :class:`~rustworkx.PyGraph` based on whether the input graph is directed
        or not.
    :rtype: :class:`~rustworkx.PyDiGraph` or :class:`~rustworkx.PyGraph`
    """
    if graph.is_directed():
        new_graph = PyDiGraph(multigraph=graph.is_multigraph())
    else:
        new_graph = PyGraph(multigraph=graph.is_multigraph())
    nodes = list(graph.nodes)
    node_indices = dict(zip(nodes, new_graph.add_nodes_from(nodes)))
    new_graph.add_edges_from([(node_indices[x[0]], node_indices[x[1]], x[2]) for x in graph.edges(data=True)])
    if keep_attributes:
        for node, node_index in node_indices.items():
            attributes = graph.nodes[node]
            attributes['__networkx_node__'] = node
            new_graph[node_index] = attributes
    return new_graph