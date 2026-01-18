import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def merge_unify(t1, t2):
    p1 = prune(t1)
    p2 = prune(t2)
    if is_none(p1) and is_none(p2):
        return p1
    if is_none(p1):
        if is_option_type(p2):
            return p2
        else:
            return OptionType(p2)
    if is_none(p2):
        return merge_unify(p2, p1)
    if is_option_type(p1) and is_option_type(p2):
        unify(p1.types[0], p2.types[0])
        return p1
    if is_option_type(p1):
        unify(p1.types[0], p2)
        return p1
    if is_option_type(p2):
        return merge_unify(p2, p1)
    unify(p1, p2)
    return p1