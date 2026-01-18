from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def get_as_array_func(n):
    """Return the function declaration f associated with a Z3 expression of the form (_ as-array f)."""
    if z3_debug():
        _z3_assert(is_as_array(n), 'as-array Z3 expression expected.')
    return FuncDeclRef(Z3_get_as_array_func_decl(n.ctx.ref(), n.as_ast()), n.ctx)