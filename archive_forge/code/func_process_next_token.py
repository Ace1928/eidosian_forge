import re
import operator
from fractions import Fraction
import sys
def process_next_token(s):
    s = s.lstrip()
    constant, rest = parse_coefficient_function(s)
    if constant is not None:
        operand_stack.append(Polynomial.constant_polynomial(constant))
        no_operand_since_opening_parenthesis[0] = False
        return rest
    variable, rest = _parse_variable(s)
    if variable:
        operand_stack.append(Polynomial.from_variable_name(variable))
        no_operand_since_opening_parenthesis[0] = False
        return rest
    next_char, rest = (s[0], s[1:])
    if next_char in list(_operators.keys()):
        operator = next_char
        eval_preceding_operators_on_stack(operator)
        operator_stack.append(operator)
        if operator in '+-':
            if no_operand_since_opening_parenthesis[0]:
                operand_stack.append(Polynomial())
                no_operand_since_opening_parenthesis[0] = False
        return rest
    if next_char in '()':
        parenthesis = next_char
        if parenthesis == '(':
            operator_stack.append('(')
            no_operand_since_opening_parenthesis[0] = True
        else:
            eval_preceding_operators_on_stack()
            top_operator = operator_stack.pop()
            assert top_operator == '('
        return rest
    raise Exception('While parsing polynomial %s' % s)