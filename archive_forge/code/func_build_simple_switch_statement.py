from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
def build_simple_switch_statement(self, node, common_var, conditions, not_in, true_val, false_val):
    result_ref = UtilNodes.ResultRefNode(node)
    true_body = Nodes.SingleAssignmentNode(node.pos, lhs=result_ref, rhs=true_val.coerce_to(node.type, self.current_env()), first=True)
    false_body = Nodes.SingleAssignmentNode(node.pos, lhs=result_ref, rhs=false_val.coerce_to(node.type, self.current_env()), first=True)
    if not_in:
        true_body, false_body = (false_body, true_body)
    cases = [Nodes.SwitchCaseNode(pos=node.pos, conditions=conditions, body=true_body)]
    common_var = unwrap_node(common_var)
    switch_node = Nodes.SwitchStatNode(pos=node.pos, test=common_var, cases=cases, else_clause=false_body)
    replacement = UtilNodes.TempResultFromStatNode(result_ref, switch_node)
    return replacement