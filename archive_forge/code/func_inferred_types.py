from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def inferred_types(entry):
    has_none = False
    has_pyobjects = False
    types = []
    for assmt in entry.cf_assignments:
        if assmt.rhs.is_none:
            has_none = True
        else:
            rhs_type = assmt.inferred_type
            if rhs_type and rhs_type.is_pyobject:
                has_pyobjects = True
            types.append(rhs_type)
    if has_none and (not has_pyobjects):
        types.append(py_object_type)
    return types