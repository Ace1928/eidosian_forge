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
def align_argument_type(self, env, arg):
    directive_locals = self.directive_locals
    orig_type = arg.type
    if arg.name in directive_locals:
        type_node = directive_locals[arg.name]
        other_type = type_node.analyse_as_type(env)
    elif isinstance(arg, CArgDeclNode) and arg.annotation and env.directives['annotation_typing']:
        type_node = arg.annotation
        other_type = arg.inject_type_from_annotations(env)
        if other_type is None:
            return arg
    else:
        return arg
    if other_type is None:
        error(type_node.pos, 'Not a type')
    elif orig_type is not py_object_type and (not orig_type.same_as(other_type)):
        error(arg.base_type.pos, 'Signature does not agree with previous declaration')
        error(type_node.pos, 'Previous declaration here')
    else:
        arg.type = other_type
        if arg.type.is_complex:
            arg.type.create_declaration_utility_code(env)
    return arg