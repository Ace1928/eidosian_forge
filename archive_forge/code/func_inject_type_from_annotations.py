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
def inject_type_from_annotations(self, env):
    annotation = self.annotation
    if not annotation:
        return None
    modifiers, arg_type = annotation.analyse_type_annotation(env, assigned_value=self.default)
    if arg_type is not None:
        self.base_type = CAnalysedBaseTypeNode(annotation.pos, type=arg_type, is_arg=True)
    if arg_type:
        if 'typing.Optional' in modifiers:
            arg_type = arg_type.resolve()
            if arg_type and (not arg_type.can_be_optional()):
                pass
            else:
                self.or_none = True
        elif arg_type is py_object_type:
            self.or_none = True
        elif self.default and self.default.is_none and (arg_type.can_be_optional() or arg_type.equivalent_type):
            if not arg_type.can_be_optional():
                arg_type = arg_type.equivalent_type
            if not self.or_none:
                warning(self.pos, "PEP-484 recommends 'typing.Optional[...]' for arguments that can be None.")
                self.or_none = True
        elif not self.or_none and arg_type.can_be_optional():
            self.not_none = True
    return arg_type