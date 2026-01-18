from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
def analyse_type_annotation(self, env, assigned_value=None):
    if self.untyped:
        return ([], None)
    annotation = self.expr
    explicit_pytype = explicit_ctype = False
    if annotation.is_dict_literal:
        warning(annotation.pos, "Dicts should no longer be used as type annotations. Use 'cython.int' etc. directly.", level=1)
        for name, value in annotation.key_value_pairs:
            if not name.is_string_literal:
                continue
            if name.value in ('type', b'type'):
                explicit_pytype = True
                if not explicit_ctype:
                    annotation = value
            elif name.value in ('ctype', b'ctype'):
                explicit_ctype = True
                annotation = value
        if explicit_pytype and explicit_ctype:
            warning(annotation.pos, 'Duplicate type declarations found in signature annotation', level=1)
    elif isinstance(annotation, TupleNode):
        warning(annotation.pos, "Tuples cannot be declared as simple tuples of types. Use 'tuple[type1, type2, ...]'.", level=1)
        return ([], None)
    with env.new_c_type_context(in_c_type_context=explicit_ctype):
        arg_type = annotation.analyse_as_type(env)
        if arg_type is None:
            self._warn_on_unknown_annotation(env, annotation)
            return ([], arg_type)
        if annotation.is_string_literal:
            warning(annotation.pos, "Strings should no longer be used for type declarations. Use 'cython.int' etc. directly.", level=1)
        if explicit_pytype and (not explicit_ctype) and (not (arg_type.is_pyobject or arg_type.equivalent_type)):
            warning(annotation.pos, 'Python type declaration in signature annotation does not refer to a Python type')
        if arg_type.is_complex:
            arg_type.create_declaration_utility_code(env)
        modifiers = annotation.analyse_pytyping_modifiers(env) if annotation.is_subscript else []
    return (modifiers, arg_type)