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
def generate_optional_kwonly_args_unpacking_code(self, all_args, code):
    optional_args = []
    first_optional_arg = -1
    num_posonly_args = 0
    for i, arg in enumerate(all_args):
        if arg.pos_only:
            num_posonly_args += 1
        if not arg.kw_only or not arg.default:
            continue
        if not optional_args:
            first_optional_arg = i
        optional_args.append(arg.name)
    if num_posonly_args > 0:
        posonly_correction = '-%d' % num_posonly_args
    else:
        posonly_correction = ''
    if optional_args:
        if len(optional_args) > 1:
            code.putln('if (kw_args > 0 && %s(kw_args <= %d)) {' % (not self.starstar_arg and 'likely' or '', len(optional_args)))
            code.putln('Py_ssize_t index;')
            code.putln('for (index = %d; index < %d && kw_args > 0; index++) {' % (first_optional_arg, first_optional_arg + len(optional_args)))
        else:
            code.putln('if (kw_args == 1) {')
            code.putln('const Py_ssize_t index = %d;' % first_optional_arg)
        code.putln('PyObject* value = __Pyx_GetKwValue_%s(%s, %s, *%s[index%s]);' % (self.signature.fastvar, Naming.kwds_cname, Naming.kwvalues_cname, Naming.pykwdlist_cname, posonly_correction))
        code.putln('if (value) { values[index] = __Pyx_Arg_NewRef_%s(value); kw_args--; }' % self.signature.fastvar)
        code.putln('else if (unlikely(PyErr_Occurred())) %s' % code.error_goto(self.pos))
        if len(optional_args) > 1:
            code.putln('}')
        code.putln('}')