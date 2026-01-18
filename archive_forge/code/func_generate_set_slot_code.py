from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def generate_set_slot_code(self, value, scope, code):
    if value == '0':
        return
    if scope.parent_type.typeptr_cname:
        target = '%s->%s' % (scope.parent_type.typeptr_cname, self.slot_name)
    else:
        assert scope.parent_type.typeobj_cname
        target = '%s.%s' % (scope.parent_type.typeobj_cname, self.slot_name)
    code.putln('%s = %s;' % (target, value))