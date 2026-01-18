import ast
import operator as op
import math
from numpy import int64
def eval_expression(expression, param_dct=dict()):
    """Parse a mathematical expression,

    Replaces variables with the values in param_dict and solves the expression

    """
    if not isinstance(expression, str):
        raise TypeError('The expression must be a string')
    if len(expression) > 10000.0:
        raise ValueError('The expression is too long.')
    expression_rep = expression.strip()
    if '()' in expression_rep:
        raise ValueError('Invalid operation in expression')
    for key, val in param_dct.items():
        expression_rep = expression_rep.replace(key, str(val))
    return _eval(ast.parse(expression_rep, mode='eval').body)