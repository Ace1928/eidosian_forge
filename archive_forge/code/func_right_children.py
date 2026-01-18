import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def right_children(self, node_index):
    """
        Returns the number of right children under the node specified
        by the given address.
        """
    children = chain.from_iterable(self.nodes[node_index]['deps'].values())
    index = self.nodes[node_index]['address']
    return sum((1 for c in children if c > index))