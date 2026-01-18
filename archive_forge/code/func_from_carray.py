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
@classmethod
def from_carray(cls, src_node, env):
    """
        Given a C array type, return a CythonArrayNode
        """
    pos = src_node.pos
    base_type = src_node.type
    none_node = NoneNode(pos)
    axes = []
    while base_type.is_array:
        axes.append(SliceNode(pos, start=none_node, stop=none_node, step=none_node))
        base_type = base_type.base_type
    axes[-1].step = IntNode(pos, value='1', is_c_literal=True)
    memslicenode = Nodes.MemoryViewSliceTypeNode(pos, axes=axes, base_type_node=base_type)
    result = CythonArrayNode(pos, base_type_node=memslicenode, operand=src_node, array_dtype=base_type)
    result = result.analyse_types(env)
    return result