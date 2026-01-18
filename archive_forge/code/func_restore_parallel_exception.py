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
def restore_parallel_exception(self, code):
    """Re-raise a parallel exception"""
    code.begin_block()
    code.put_ensure_gil(declare_gilstate=True)
    code.put_giveref(Naming.parallel_exc_type, py_object_type)
    code.putln('__Pyx_ErrRestoreWithState(%s, %s, %s);' % self.parallel_exc)
    pos_info = chain(*zip(self.pos_info, self.parallel_pos_info))
    code.putln('%s = %s; %s = %s; %s = %s;' % tuple(pos_info))
    code.put_release_ensured_gil()
    code.end_block()