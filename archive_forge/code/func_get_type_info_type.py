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
def get_type_info_type(self, env):
    env_module = env
    while not env_module.is_module_scope:
        env_module = env_module.outer_scope
    typeinfo_module = env_module.find_module('libcpp.typeinfo', self.pos)
    typeinfo_entry = typeinfo_module.lookup('type_info')
    return PyrexTypes.CFakeReferenceType(PyrexTypes.c_const_type(typeinfo_entry.type))