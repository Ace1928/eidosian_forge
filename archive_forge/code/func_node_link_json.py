import sys
import functools
from .rustworkx import *
import rustworkx.visit
@functools.singledispatch
def node_link_json(graph, path=None, graph_attrs=None, node_attrs=None, edge_attrs=None):
    """Generate a JSON object representing a graph in a node-link format

    :param graph: The graph to generate the JSON for. Can either be a
        :class:`~retworkx.PyGraph` or :class:`~retworkx.PyDiGraph`.
    :param str path: An optional path to write the JSON output to. If specified
        the function will not return anything and instead will write the JSON
        to the file specified.
    :param graph_attrs: An optional callable that will be passed the
        :attr:`~.PyGraph.attrs` attribute of the graph and is expected to
        return a dictionary of string keys to string values representing the
        graph attributes. This dictionary will be included as attributes in
        the output JSON. If anything other than a dictionary with string keys
        and string values is returned an exception will be raised.
    :param node_attrs: An optional callable that will be passed the node data
        payload for each node in the graph and is expected to return a
        dictionary of string keys to string values representing the data payload.
        This dictionary will be used as the ``data`` field for each node.
    :param edge_attrs:  An optional callable that will be passed the edge data
        payload for each node in the graph and is expected to return a
        dictionary of string keys to string values representing the data payload.
        This dictionary will be used as the ``data`` field for each edge.

    :returns: Either the JSON string for the payload or ``None`` if ``path`` is specified
    :rtype: str
    """
    raise TypeError('Invalid Input Type %s for graph' % type(graph))