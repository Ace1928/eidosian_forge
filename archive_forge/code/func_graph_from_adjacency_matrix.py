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
def graph_from_adjacency_matrix(matrix, node_prefix='', directed=False):
    """Creates a basic graph out of an adjacency matrix.

    The matrix has to be a list of rows of values
    representing an adjacency matrix.
    The values can be anything: bool, int, float, as long
    as they can evaluate to True or False.
    """
    node_orig = 1
    if directed:
        graph = Dot(graph_type='digraph')
    else:
        graph = Dot(graph_type='graph')
    for row in matrix:
        if not directed:
            skip = matrix.index(row)
            r = row[skip:]
        else:
            skip = 0
            r = row
        node_dest = skip + 1
        for e in r:
            if e:
                graph.add_edge(Edge('%s%s' % (node_prefix, node_orig), '%s%s' % (node_prefix, node_dest)))
            node_dest += 1
        node_orig += 1
    return graph