from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def find_spanning_type(type1, type2):
    if type1 is type2:
        result_type = type1
    elif type1 is PyrexTypes.c_bint_type or type2 is PyrexTypes.c_bint_type:
        return py_object_type
    else:
        result_type = PyrexTypes.spanning_type(type1, type2)
    if result_type in (PyrexTypes.c_double_type, PyrexTypes.c_float_type, Builtin.float_type):
        return PyrexTypes.c_double_type
    return result_type