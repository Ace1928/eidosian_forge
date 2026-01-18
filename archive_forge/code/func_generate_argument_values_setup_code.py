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
def generate_argument_values_setup_code(self, args, code, decl_code):
    max_args = len(args)
    decl_code.putln('PyObject* values[%d] = {%s};' % (max_args, ','.join('0' * max_args)))
    if self.target.defaults_struct:
        code.putln('%s *%s = __Pyx_CyFunction_Defaults(%s, %s);' % (self.target.defaults_struct, Naming.dynamic_args_cname, self.target.defaults_struct, Naming.self_cname))
    for i, arg in enumerate(args):
        if arg.default and arg.type.is_pyobject:
            default_value = arg.calculate_default_value_code(code)
            code.putln('values[%d] = __Pyx_Arg_NewRef_%s(%s);' % (i, self.signature.fastvar, arg.type.as_pyobject(default_value)))