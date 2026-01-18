from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
@staticmethod
def strict_eq(f0, f1):
    if f0 == f1:
        return True
    try:
        return isnan(f0) and isnan(f1)
    except TypeError:
        return False