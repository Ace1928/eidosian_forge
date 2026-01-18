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
def analyse_buffer_index(self, env, getting):
    if is_pythran_expr(self.base.type):
        index_with_type_list = [(idx, idx.type) for idx in self.indices]
        self.type = PythranExpr(pythran_indexing_type(self.base.type, index_with_type_list))
    else:
        self.base = self.base.coerce_to_simple(env)
        self.type = self.base.type.dtype
    self.buffer_type = self.base.type
    if getting and (self.type.is_pyobject or self.type.is_pythran_expr):
        self.is_temp = True