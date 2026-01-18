from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def needs_assignment_synthesis(self, env, code=None):
    if self.is_staticmethod:
        return True
    if self.specialized_cpdefs or self.entry.is_fused_specialized:
        return False
    if self.no_assignment_synthesis:
        return False
    if self.entry.is_special:
        return False
    if self.entry.is_anonymous:
        return True
    if env.is_module_scope or env.is_c_class_scope:
        if code is None:
            return self.local_scope.directives['binding']
        else:
            return code.globalstate.directives['binding']
    return env.is_py_class_scope or env.is_closure_scope