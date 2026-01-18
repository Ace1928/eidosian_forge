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
def generate_argument_declarations(self, env, code):
    for arg in self.args:
        if arg.is_generic:
            if arg.needs_conversion:
                code.putln('PyObject *%s = 0;' % arg.hdr_cname)
            else:
                code.put_var_declaration(arg.entry)
    for entry in env.var_entries:
        if entry.is_arg:
            code.put_var_declaration(entry)
    if self.signature_has_generic_args():
        nargs_code = 'CYTHON_UNUSED Py_ssize_t %s;' % Naming.nargs_cname
        if self.signature.use_fastcall:
            code.putln('#if !CYTHON_METH_FASTCALL')
            code.putln(nargs_code)
            code.putln('#endif')
        else:
            code.putln(nargs_code)
    code.putln('CYTHON_UNUSED PyObject *const *%s;' % Naming.kwvalues_cname)