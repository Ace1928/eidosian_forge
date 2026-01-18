import random
import sys
from . import Nodes
def is_bifurcating(self, node=None):
    """Return True if tree downstream of node is strictly bifurcating."""
    if node is None:
        node = self.root
    if node == self.root and len(self.node(node).succ) == 3:
        return self.is_bifurcating(self.node(node).succ[0]) and self.is_bifurcating(self.node(node).succ[1]) and self.is_bifurcating(self.node(node).succ[2])
    if len(self.node(node).succ) == 2:
        return self.is_bifurcating(self.node(node).succ[0]) and self.is_bifurcating(self.node(node).succ[1])
    elif len(self.node(node).succ) == 0:
        return True
    else:
        return False