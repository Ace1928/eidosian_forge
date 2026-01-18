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
def analyse_as_c_array(self, env, is_slice):
    base_type = self.base.type
    self.type = base_type.base_type
    if self.type.is_cpp_class:
        self.type = PyrexTypes.CReferenceType(self.type)
    if is_slice:
        self.type = base_type
    elif self.index.type.is_pyobject:
        self.index = self.index.coerce_to(PyrexTypes.c_py_ssize_t_type, env)
    elif not self.index.type.is_int:
        error(self.pos, "Invalid index type '%s'" % self.index.type)
    return self