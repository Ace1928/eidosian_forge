from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
def search(self, node):
    """ Facility to get values of the matcher for a given node. """
    self.visit(node)
    return self.result