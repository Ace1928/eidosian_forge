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
def generate_stararg_init_code(self, max_positional_args, code):
    if self.starstar_arg:
        self.starstar_arg.entry.xdecref_cleanup = 0
        code.putln('%s = PyDict_New(); if (unlikely(!%s)) return %s;' % (self.starstar_arg.entry.cname, self.starstar_arg.entry.cname, self.error_value()))
        code.put_var_gotref(self.starstar_arg.entry)
    if self.star_arg:
        self.star_arg.entry.xdecref_cleanup = 0
        if max_positional_args == 0:
            assert not self.signature.use_fastcall
            code.put_incref(Naming.args_cname, py_object_type)
            code.putln('%s = %s;' % (self.star_arg.entry.cname, Naming.args_cname))
        else:
            code.putln('%s = __Pyx_ArgsSlice_%s(%s, %d, %s);' % (self.star_arg.entry.cname, self.signature.fastvar, Naming.args_cname, max_positional_args, Naming.nargs_cname))
            code.putln('if (unlikely(!%s)) {' % self.star_arg.entry.type.nullcheck_string(self.star_arg.entry.cname))
            if self.starstar_arg:
                code.put_var_decref_clear(self.starstar_arg.entry)
            code.put_finish_refcount_context()
            code.putln('return %s;' % self.error_value())
            code.putln('}')
            code.put_var_gotref(self.star_arg.entry)