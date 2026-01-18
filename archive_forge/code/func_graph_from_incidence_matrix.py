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
def graph_from_incidence_matrix(matrix, node_prefix='', directed=False):
    """Creates a basic graph out of an incidence matrix.

    The matrix has to be a list of rows of values
    representing an incidence matrix.
    The values can be anything: bool, int, float, as long
    as they can evaluate to True or False.
    """
    if directed:
        graph = Dot(graph_type='digraph')
    else:
        graph = Dot(graph_type='graph')
    for row in matrix:
        nodes = []
        c = 1
        for node in row:
            if node:
                nodes.append(c * node)
            c += 1
            nodes.sort()
        if len(nodes) == 2:
            graph.add_edge(Edge('%s%s' % (node_prefix, abs(nodes[0])), '%s%s' % (node_prefix, nodes[1])))
    if not directed:
        graph.set_simplify(True)
    return graph