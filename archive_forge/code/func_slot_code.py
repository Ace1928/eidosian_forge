from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def slot_code(self, scope):
    dict_entry = scope.lookup_here('__dict__') if not scope.is_closure_class_scope else None
    if dict_entry and dict_entry.is_variable:
        if getattr(dict_entry.type, 'cname', None) != 'PyDict_Type':
            error(dict_entry.pos, "__dict__ slot must be of type 'dict'")
            return '0'
        type = scope.parent_type
        if type.typedef_flag:
            objstruct = type.objstruct_cname
        else:
            objstruct = 'struct %s' % type.objstruct_cname
        return 'offsetof(%s, %s)' % (objstruct, dict_entry.cname)
    else:
        return '0'