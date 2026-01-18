import re
import warnings
from enum import Enum
from math import gcd
def polynomial_atoms(self):
    """Return a set of expressions used as atoms in polynomial self.
        """
    found = set()

    def visit(expr, found=found):
        if expr.op is Op.FACTORS:
            for b in expr.data:
                b.traverse(visit)
            return expr
        if expr.op in (Op.TERMS, Op.COMPLEX):
            return
        if expr.op is Op.APPLY and isinstance(expr.data[0], ArithOp):
            if expr.data[0] is ArithOp.POW:
                expr.data[1][0].traverse(visit)
                return expr
            return
        if expr.op in (Op.INTEGER, Op.REAL):
            return expr
        found.add(expr)
        if expr.op in (Op.INDEXING, Op.APPLY):
            return expr
    self.traverse(visit)
    return found