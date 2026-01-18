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
def analyse_pytyping_modifiers(self, env):
    modifiers = []
    modifier_node = self
    while modifier_node.is_templated_type_node and modifier_node.base_type_node and (len(modifier_node.positional_args) == 1):
        modifier_type = self.base_type_node.analyse_as_type(env)
        if modifier_type.python_type_constructor_name and modifier_type.modifier_name:
            modifiers.append(modifier_type.modifier_name)
        modifier_node = modifier_node.positional_args[0]
    return modifiers