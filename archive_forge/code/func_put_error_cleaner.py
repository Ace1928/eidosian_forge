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
def put_error_cleaner(self, code, exc_vars):
    code.globalstate.use_utility_code(reset_exception_utility_code)
    if self.is_try_finally_in_nogil:
        code.put_ensure_gil(declare_gilstate=False)
        code.putln('__Pyx_PyThreadState_assign')
    code.putln('if (PY_MAJOR_VERSION >= 3) {')
    for var in exc_vars[3:]:
        code.put_xgiveref(var, py_object_type)
    code.putln('__Pyx_ExceptionReset(%s, %s, %s);' % exc_vars[3:])
    code.putln('}')
    for var in exc_vars[:3]:
        code.put_xdecref_clear(var, py_object_type)
    if self.is_try_finally_in_nogil:
        code.put_release_ensured_gil()
    code.putln(' '.join(['%s = 0;'] * 3) % exc_vars[3:])