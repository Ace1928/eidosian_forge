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
def end_parallel_block(self, code):
    """
        To ensure all OpenMP threads have thread states, we ensure the GIL
        in each thread (which creates a thread state if it doesn't exist),
        after which we release the GIL.
        On exit, reacquire the GIL and release the thread state.

        If compiled without OpenMP support (at the C level), then we still have
        to acquire the GIL to decref any object temporaries.
        """
    begin_code = self.begin_of_parallel_block
    self.begin_of_parallel_block = None
    if self.error_label_used:
        end_code = code
        begin_code.putln('#ifdef _OPENMP')
        begin_code.put_ensure_gil(declare_gilstate=True)
        begin_code.putln('Py_BEGIN_ALLOW_THREADS')
        begin_code.putln('#endif /* _OPENMP */')
        end_code.putln('#ifdef _OPENMP')
        end_code.putln('Py_END_ALLOW_THREADS')
        end_code.putln('#else')
        end_code.put_safe('{\n')
        end_code.put_ensure_gil()
        end_code.putln('#endif /* _OPENMP */')
        self.cleanup_temps(end_code)
        end_code.put_release_ensured_gil()
        end_code.putln('#ifndef _OPENMP')
        end_code.put_safe('}\n')
        end_code.putln('#endif /* _OPENMP */')