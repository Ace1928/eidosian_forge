import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def contains_address(self, node_address):
    """
        Returns true if the graph contains a node with the given node
        address, false otherwise.
        """
    return node_address in self.nodes