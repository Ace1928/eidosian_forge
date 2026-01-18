import random
import sys
from . import Nodes
def is_preterminal(self, node):
    """Return True if all successors of a node are terminal ones."""
    if self.is_terminal(node):
        return False not in [self.is_terminal(n) for n in self.node(node).succ]
    else:
        return False