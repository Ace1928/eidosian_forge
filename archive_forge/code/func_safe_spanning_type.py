from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def safe_spanning_type(types, might_overflow, scope):
    result_type = simply_type(reduce(find_spanning_type, types))
    if result_type.is_pyobject:
        if result_type.name == 'str':
            return py_object_type
        else:
            return result_type
    elif result_type is PyrexTypes.c_double_type or result_type is PyrexTypes.c_float_type:
        return result_type
    elif result_type is PyrexTypes.c_bint_type:
        return result_type
    elif result_type.is_pythran_expr:
        return result_type
    elif result_type.is_ptr:
        return result_type
    elif result_type.is_cpp_class:
        return result_type
    elif result_type.is_struct:
        return result_type
    elif result_type.is_memoryviewslice:
        return result_type
    elif result_type is PyrexTypes.soft_complex_type:
        return result_type
    elif result_type == PyrexTypes.c_double_complex_type:
        return result_type
    elif (result_type.is_int or result_type.is_enum) and (not might_overflow):
        return result_type
    elif not result_type.can_coerce_to_pyobject(scope) and (not result_type.is_error):
        return result_type
    return py_object_type