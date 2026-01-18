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
def analyse_argument_types(self, env):
    self.directive_locals = env.directives.get('locals', {})
    allow_none_for_extension_args = env.directives['allow_none_for_extension_args']
    f2s = env.fused_to_specific
    env.fused_to_specific = None
    for arg in self.args:
        if hasattr(arg, 'name'):
            name_declarator = None
        else:
            base_type = arg.base_type.analyse(env)
            if has_np_pythran(env) and base_type.is_pythran_expr:
                base_type = PyrexTypes.FusedType([base_type, base_type.org_buffer])
            name_declarator, type = arg.declarator.analyse(base_type, env)
            arg.name = name_declarator.name
            arg.type = type
        self.align_argument_type(env, arg)
        if name_declarator and name_declarator.cname:
            error(self.pos, 'Python function argument cannot have C name specification')
        arg.type = arg.type.as_argument_type()
        arg.hdr_type = None
        arg.needs_conversion = 0
        arg.needs_type_test = 0
        arg.is_generic = 1
        if arg.type.is_pyobject or arg.type.is_buffer or arg.type.is_memoryviewslice:
            if arg.or_none:
                arg.accept_none = True
            elif arg.not_none:
                arg.accept_none = False
            elif arg.type.is_extension_type or arg.type.is_builtin_type or arg.type.is_buffer or arg.type.is_memoryviewslice:
                if arg.default and arg.default.constant_result is None:
                    arg.accept_none = True
                else:
                    arg.accept_none = allow_none_for_extension_args
            else:
                arg.accept_none = True
        elif not arg.type.is_error:
            arg.accept_none = True
            if arg.not_none:
                error(arg.pos, "Only Python type arguments can have 'not None'")
            if arg.or_none:
                error(arg.pos, "Only Python type arguments can have 'or None'")
        if arg.type.is_fused:
            self.has_fused_arguments = True
    env.fused_to_specific = f2s
    if has_np_pythran(env):
        self.np_args_idx = [i for i, a in enumerate(self.args) if a.type.is_numpy_buffer]
    else:
        self.np_args_idx = []