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
def allow_none(node, default_value, env):
    from .UtilNodes import EvalWithTempExprNode, ResultRefNode
    node_ref = ResultRefNode(node)
    new_expr = CondExprNode(node.pos, true_val=IntNode(node.pos, type=c_int, value=default_value, constant_result=int(default_value) if default_value.isdigit() else not_a_constant), false_val=node_ref.coerce_to(c_int, env), test=PrimaryCmpNode(node.pos, operand1=node_ref, operator='is', operand2=NoneNode(node.pos)).analyse_types(env)).analyse_result_type(env)
    return EvalWithTempExprNode(node_ref, new_expr)