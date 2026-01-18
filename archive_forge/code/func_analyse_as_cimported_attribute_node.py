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
def analyse_as_cimported_attribute_node(self, env, target):
    module_scope = self.obj.analyse_as_module(env)
    if module_scope:
        entry = module_scope.lookup_here(self.attribute)
        if entry and (not entry.known_standard_library_import) and (entry.is_cglobal or entry.is_cfunction or entry.is_type or entry.is_const):
            return self.as_name_node(env, entry, target)
        if self.is_cimported_module_without_shadow(env):
            error(self.pos, "cimported module has no attribute '%s'" % self.attribute)
            return self
    return None