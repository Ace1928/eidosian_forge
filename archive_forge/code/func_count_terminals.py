import random
import sys
from . import Nodes
def count_terminals(self, node=None):
    """Count the number of terminal nodes that are attached to a node."""
    if node is None:
        node = self.root
    return len([n for n in self._walk(node) if self.is_terminal(n)])