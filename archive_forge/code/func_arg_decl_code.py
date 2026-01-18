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
def arg_decl_code(arg):
    entry = arg.entry
    if entry.in_closure:
        cname = entry.original_cname
    else:
        cname = entry.cname
    decl = entry.type.declaration_code(cname)
    if not entry.cf_used:
        decl = 'CYTHON_UNUSED ' + decl
    return decl