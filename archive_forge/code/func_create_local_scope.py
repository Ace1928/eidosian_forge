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
def create_local_scope(self, env):
    genv = env
    while genv.is_py_class_scope or genv.is_c_class_scope:
        genv = genv.outer_scope
    if self.needs_closure:
        cls = GeneratorExpressionScope if self.is_generator_expression else ClosureScope
        lenv = cls(name=self.entry.name, outer_scope=genv, parent_scope=env, scope_name=self.entry.cname)
    else:
        lenv = LocalScope(name=self.entry.name, outer_scope=genv, parent_scope=env)
    lenv.return_type = self.return_type
    type = self.entry.type
    if type.is_cfunction:
        lenv.nogil = type.nogil and (not type.with_gil)
    self.local_scope = lenv
    lenv.directives = env.directives
    return lenv