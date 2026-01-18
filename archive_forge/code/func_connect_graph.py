import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def connect_graph(self):
    """
        Fully connects all non-root nodes.  All nodes are set to be dependents
        of the root node.
        """
    for node1 in self.nodes.values():
        for node2 in self.nodes.values():
            if node1['address'] != node2['address'] and node2['rel'] != 'TOP':
                relation = node2['rel']
                node1['deps'].setdefault(relation, [])
                node1['deps'][relation].append(node2['address'])