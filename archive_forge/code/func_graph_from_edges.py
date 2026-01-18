import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def graph_from_edges(edge_list, node_prefix='', directed=False):
    """Creates a basic graph out of an edge list.

    The edge list has to be a list of tuples representing
    the nodes connected by the edge.
    The values can be anything: bool, int, float, str.

    If the graph is undirected by default, it is only
    calculated from one of the symmetric halves of the matrix.
    """
    if directed:
        graph = Dot(graph_type='digraph')
    else:
        graph = Dot(graph_type='graph')
    for edge in edge_list:
        if isinstance(edge[0], str):
            src = node_prefix + edge[0]
        else:
            src = node_prefix + str(edge[0])
        if isinstance(edge[1], str):
            dst = node_prefix + edge[1]
        else:
            dst = node_prefix + str(edge[1])
        e = Edge(src, dst)
        graph.add_edge(e)
    return graph