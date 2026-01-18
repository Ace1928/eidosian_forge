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
def implicit_multiplication(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT) -> List[TOKEN]:
    """Makes the multiplication operator optional in most cases.

    Use this before :func:`implicit_application`, otherwise expressions like
    ``sin 2x`` will be parsed as ``x * sin(2)`` rather than ``sin(2*x)``.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_multiplication)
    >>> transformations = standard_transformations + (implicit_multiplication,)
    >>> parse_expr('3 x y', transformations=transformations)
    3*x*y
    """
    res1 = _group_parentheses(implicit_multiplication)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _implicit_multiplication(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result