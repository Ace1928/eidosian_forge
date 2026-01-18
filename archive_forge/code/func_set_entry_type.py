from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def set_entry_type(self, entry, entry_type, scope):
    for e in entry.all_entries():
        e.type = entry_type
        if e.type.is_memoryviewslice:
            e.init = e.type.default_value
        if e.type.is_cpp_class:
            if scope.directives['cpp_locals']:
                e.make_cpp_optional()
            else:
                e.type.check_nullary_constructor(entry.pos)