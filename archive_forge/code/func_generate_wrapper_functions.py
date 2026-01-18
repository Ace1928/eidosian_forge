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
def generate_wrapper_functions(self, code):
    k = 0
    entry = self.entry
    func_type = entry.type
    while entry.prev_entry is not None:
        k += 1
        entry = entry.prev_entry
        entry.func_cname = '%s%swrap_%s' % (self.entry.func_cname, Naming.pyrex_prefix, k)
        code.putln()
        self.generate_function_header(code, 0, with_dispatch=entry.type.is_overridable, with_opt_args=entry.type.optional_arg_count, cname=entry.func_cname)
        if not self.return_type.is_void:
            code.put('return ')
        args = self.type.args
        arglist = [arg.cname for arg in args[:len(args) - self.type.optional_arg_count]]
        if entry.type.is_overridable:
            arglist.append(Naming.skip_dispatch_cname)
        elif func_type.is_overridable:
            arglist.append('0')
        if entry.type.optional_arg_count:
            arglist.append(Naming.optional_args_cname)
        elif func_type.optional_arg_count:
            arglist.append('NULL')
        code.putln('%s(%s);' % (self.entry.func_cname, ', '.join(arglist)))
        code.putln('}')