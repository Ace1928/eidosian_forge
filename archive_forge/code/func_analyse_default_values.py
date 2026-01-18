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
def analyse_default_values(self, env):
    default_seen = 0
    for arg in self.args:
        if arg.default:
            default_seen = 1
            if arg.is_generic:
                arg.default = arg.default.analyse_types(env)
                arg.default = arg.default.coerce_to(arg.type, env)
            elif arg.is_special_method_optional:
                if not arg.default.is_none:
                    error(arg.pos, 'This argument cannot have a non-None default value')
                    arg.default = None
            else:
                error(arg.pos, 'This argument cannot have a default value')
                arg.default = None
        elif arg.kw_only:
            default_seen = 1
        elif default_seen:
            error(arg.pos, 'Non-default argument following default argument')