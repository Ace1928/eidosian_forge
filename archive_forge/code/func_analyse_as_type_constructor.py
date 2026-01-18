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
def analyse_as_type_constructor(self, env):
    type = self.function.analyse_as_type(env)
    if type and type.is_struct_or_union:
        args, kwds = self.explicit_args_kwds()
        items = []
        for arg, member in zip(args, type.scope.var_entries):
            items.append(DictItemNode(pos=arg.pos, key=StringNode(pos=arg.pos, value=member.name), value=arg))
        if kwds:
            items += kwds.key_value_pairs
        self.key_value_pairs = items
        self.__class__ = DictNode
        self.analyse_types(env)
        self.coerce_to(type, env)
        return True
    elif type and type.is_cpp_class:
        self.args = [arg.analyse_types(env) for arg in self.args]
        constructor = type.scope.lookup('<init>')
        if not constructor:
            error(self.function.pos, "no constructor found for C++  type '%s'" % self.function.name)
            self.type = error_type
            return self
        self.function = RawCNameExprNode(self.function.pos, constructor.type)
        self.function.entry = constructor
        self.function.set_cname(type.empty_declaration_code())
        self.analyse_c_function_call(env)
        self.type = type
        return True