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
def extract_conditions(self, cond, allow_not_in):
    while True:
        if isinstance(cond, (ExprNodes.CoerceToTempNode, ExprNodes.CoerceToBooleanNode)):
            cond = cond.arg
        elif isinstance(cond, ExprNodes.BoolBinopResultNode):
            cond = cond.arg.arg
        elif isinstance(cond, UtilNodes.EvalWithTempExprNode):
            cond = cond.subexpression
        elif isinstance(cond, ExprNodes.TypecastNode):
            cond = cond.operand
        else:
            break
    if isinstance(cond, ExprNodes.PrimaryCmpNode):
        if cond.cascade is not None:
            return self.NO_MATCH
        elif cond.is_c_string_contains() and isinstance(cond.operand2, (ExprNodes.UnicodeNode, ExprNodes.BytesNode)):
            not_in = cond.operator == 'not_in'
            if not_in and (not allow_not_in):
                return self.NO_MATCH
            if isinstance(cond.operand2, ExprNodes.UnicodeNode) and cond.operand2.contains_surrogates():
                return self.NO_MATCH
            return (not_in, cond.operand1, self.extract_in_string_conditions(cond.operand2))
        elif not cond.is_python_comparison():
            if cond.operator == '==':
                not_in = False
            elif allow_not_in and cond.operator == '!=':
                not_in = True
            else:
                return self.NO_MATCH
            if is_common_value(cond.operand1, cond.operand1):
                if cond.operand2.is_literal:
                    return (not_in, cond.operand1, [cond.operand2])
                elif getattr(cond.operand2, 'entry', None) and cond.operand2.entry.is_const:
                    return (not_in, cond.operand1, [cond.operand2])
            if is_common_value(cond.operand2, cond.operand2):
                if cond.operand1.is_literal:
                    return (not_in, cond.operand2, [cond.operand1])
                elif getattr(cond.operand1, 'entry', None) and cond.operand1.entry.is_const:
                    return (not_in, cond.operand2, [cond.operand1])
    elif isinstance(cond, ExprNodes.BoolBinopNode):
        if cond.operator == 'or' or (allow_not_in and cond.operator == 'and'):
            allow_not_in = cond.operator == 'and'
            not_in_1, t1, c1 = self.extract_conditions(cond.operand1, allow_not_in)
            not_in_2, t2, c2 = self.extract_conditions(cond.operand2, allow_not_in)
            if t1 is not None and not_in_1 == not_in_2 and is_common_value(t1, t2):
                if not not_in_1 or allow_not_in:
                    return (not_in_1, t1, c1 + c2)
    return self.NO_MATCH