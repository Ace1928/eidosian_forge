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
def prepare_argument_coercion(self, env):
    for arg in self.args:
        if not arg.type.is_pyobject:
            if not arg.type.create_from_py_utility_code(env):
                pass
        elif arg.hdr_type and (not arg.hdr_type.is_pyobject):
            if not arg.hdr_type.create_to_py_utility_code(env):
                pass
    if self.starstar_arg and (not self.starstar_arg.entry.cf_used):
        entry = self.starstar_arg.entry
        entry.xdecref_cleanup = 1
        for ass in entry.cf_assignments:
            if not ass.is_arg and ass.lhs.is_name:
                ass.lhs.cf_maybe_null = True