import ast
from dataclasses import dataclass
import operator as op
def eval_expr(expr):
    """
    >>> eval_expr('2*6')
    12
    >>> eval_expr('2**6')
    64
    >>> eval_expr('1 + 2*3**(4) / (6 + -7)')
    -161.0
    """
    try:
        return eval_(ast.parse(expr, mode='eval').body)
    except (TypeError, SyntaxError, KeyError) as e:
        raise ValueError(f'{expr!r} is not a valid or supported arithmetic expression.') from e