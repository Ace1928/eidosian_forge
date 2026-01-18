from tokenize import (generate_tokens, untokenize, TokenError,
from keyword import iskeyword
import ast
import unicodedata
from io import StringIO
import builtins
import types
from typing import Tuple as tTuple, Dict as tDict, Any, Callable, \
from sympy.assumptions.ask import AssumptionKeys
from sympy.core.basic import Basic
from sympy.core import Symbol
from sympy.core.function import Function
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Max, Min
def function_exponentiation(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Allows functions to be exponentiated, e.g. ``cos**2(x)``.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, function_exponentiation)
    >>> transformations = standard_transformations + (function_exponentiation,)
    >>> parse_expr('sin**4(x)', transformations=transformations)
    sin(x)**4
    """
    result: List[TOKEN] = []
    exponent: List[TOKEN] = []
    consuming_exponent = False
    level = 0
    for tok, nextTok in zip(tokens, tokens[1:]):
        if tok[0] == NAME and nextTok[0] == OP and (nextTok[1] == '**'):
            if _token_callable(tok, local_dict, global_dict):
                consuming_exponent = True
        elif consuming_exponent:
            if tok[0] == NAME and tok[1] == 'Function':
                tok = (NAME, 'Symbol')
            exponent.append(tok)
            if tok[0] == nextTok[0] == OP and tok[1] == ')' and (nextTok[1] == '('):
                consuming_exponent = False
            if tok[0] == nextTok[0] == OP and tok[1] == '*' and (nextTok[1] == '('):
                consuming_exponent = False
                del exponent[-1]
            continue
        elif exponent and (not consuming_exponent):
            if tok[0] == OP:
                if tok[1] == '(':
                    level += 1
                elif tok[1] == ')':
                    level -= 1
            if level == 0:
                result.append(tok)
                result.extend(exponent)
                exponent = []
                continue
        result.append(tok)
    if tokens:
        result.append(tokens[-1])
    if exponent:
        result.extend(exponent)
    return result