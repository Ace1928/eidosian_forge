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
def declare_cpdef_wrapper(self, env):
    if not self.overridable:
        return
    if self.is_static_method:
        error(self.pos, 'static cpdef methods not yet supported')
    name = self.entry.name
    py_func_body = self.call_self_node(is_module_scope=env.is_module_scope)
    if self.is_static_method:
        from .ExprNodes import NameNode
        decorators = [DecoratorNode(self.pos, decorator=NameNode(self.pos, name=EncodedString('staticmethod')))]
        decorators[0].decorator.analyse_types(env)
    else:
        decorators = []
    self.py_func = DefNode(pos=self.pos, name=self.entry.name, args=self.args, star_arg=None, starstar_arg=None, doc=self.doc, body=py_func_body, decorators=decorators, is_wrapper=1)
    self.py_func.is_module_scope = env.is_module_scope
    self.py_func.analyse_declarations(env)
    self.py_func.entry.is_overridable = True
    self.py_func_stat = StatListNode(self.pos, stats=[self.py_func])
    self.py_func.type = PyrexTypes.py_object_type
    self.entry.as_variable = self.py_func.entry
    self.entry.used = self.entry.as_variable.used = True
    env.entries[name] = self.entry
    if not self.entry.is_final_cmethod and (not env.is_module_scope or Options.lookup_module_cpdef):
        if self.override:
            assert self.entry.is_fused_specialized
            self.override.py_func = self.py_func
        else:
            self.override = OverrideCheckNode(self.pos, py_func=self.py_func)
            self.body = StatListNode(self.pos, stats=[self.override, self.body])