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
def declare_python_arg(self, env, arg):
    if arg:
        if env.directives['infer_types'] != False:
            type = PyrexTypes.unspecified_type
        else:
            type = py_object_type
        entry = env.declare_var(arg.name, type, arg.pos)
        entry.is_arg = 1
        entry.used = 1
        entry.init = '0'
        entry.xdecref_cleanup = 1
        arg.entry = entry