from pythran.passmanager import Transformation
from pythran.analyses import Ancestors
from pythran.syntax import PythranSyntaxError
from functools import reduce
import gast as ast
@staticmethod
def match_is_none(node):
    noned_var = is_is_none(node)
    if noned_var is None:
        noned_var = is_is_not_none(node)
        negated = noned_var is not None
    else:
        negated = False
    return (noned_var, negated)