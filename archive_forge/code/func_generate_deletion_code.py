from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
def generate_deletion_code(self, code, ignore_nonexisting=False):
    self.obj.generate_evaluation_code(code)
    if self.is_py_attr or (self.entry.scope.is_property_scope and u'__del__' in self.entry.scope.entries):
        code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectSetAttrStr', 'ObjectHandling.c'))
        code.put_error_if_neg(self.pos, '__Pyx_PyObject_DelAttrStr(%s, %s)' % (self.obj.py_result(), code.intern_identifier(self.attribute)))
    else:
        error(self.pos, 'Cannot delete C attribute of extension type')
    self.obj.generate_disposal_code(code)
    self.obj.free_temps(code)