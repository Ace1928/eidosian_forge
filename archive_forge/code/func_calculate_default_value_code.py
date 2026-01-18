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
def calculate_default_value_code(self, code):
    if self.default_value is None:
        if self.default:
            if self.default.is_literal:
                self.default.generate_evaluation_code(code)
                return self.type.cast_code(self.default.result())
            self.default_value = code.get_argument_default_const(self.type)
    return self.default_value