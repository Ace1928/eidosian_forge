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
def declare_generator_body(self, env):
    prefix = env.next_id(env.scope_prefix)
    name = env.next_id('generator')
    cname = Naming.genbody_prefix + prefix + name
    entry = env.declare_var(None, py_object_type, self.pos, cname=cname, visibility='private')
    entry.func_cname = cname
    entry.qualified_name = EncodedString(self.name)
    entry.used = True
    self.entry = entry