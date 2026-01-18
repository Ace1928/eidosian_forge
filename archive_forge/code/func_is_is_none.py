from pythran.passmanager import Transformation
from pythran.analyses import Ancestors
from pythran.syntax import PythranSyntaxError
from functools import reduce
import gast as ast
def is_is_none(expr):
    if not isinstance(expr, ast.Compare):
        return None
    if len(expr.ops) != 1:
        exprs = [expr.left] + expr.comparators
        if any((is_none(expr) for expr in exprs)):
            raise PythranSyntaxError('is None in complex condition', expr)
        return None
    if not isinstance(expr.ops[0], (ast.Eq, ast.Is)):
        return None
    if is_none(expr.left):
        return expr.comparators[0]
    if is_none(expr.comparators[0]):
        return expr.left
    return None