import re
import operator
from fractions import Fraction
import sys
def eval_preceding_operators_on_stack(operator=None):
    while operator_stack:
        top_operator = operator_stack[-1]
        if top_operator == '(':
            return
        if _operator_precedence[top_operator] < _operator_precedence[operator]:
            return
        top_operator = operator_stack.pop()
        r = operand_stack.pop()
        l = operand_stack.pop()
        operand_stack.append(_apply_operator(top_operator, l, r))