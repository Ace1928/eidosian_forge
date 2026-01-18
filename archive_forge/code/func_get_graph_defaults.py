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
def get_graph_defaults(self, **attrs):
    graph_nodes = self.get_node('graph')
    if isinstance(graph_nodes, (list, tuple)):
        return [node.get_attributes() for node in graph_nodes]
    return graph_nodes.get_attributes()