import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def redirect_arcs(self, originals, redirect):
    """
        Redirects arcs to any of the nodes in the originals list
        to the redirect node address.
        """
    for node in self.nodes.values():
        new_deps = []
        for dep in node['deps']:
            if dep in originals:
                new_deps.append(redirect)
            else:
                new_deps.append(dep)
        node['deps'] = new_deps