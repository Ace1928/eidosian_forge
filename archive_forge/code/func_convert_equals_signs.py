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
def convert_equals_signs(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT) -> List[TOKEN]:
    """ Transforms all the equals signs ``=`` to instances of Eq.

    Parses the equals signs in the expression and replaces them with
    appropriate Eq instances. Also works with nested equals signs.

    Does not yet play well with function arguments.
    For example, the expression ``(x=y)`` is ambiguous and can be interpreted
    as x being an argument to a function and ``convert_equals_signs`` will not
    work for this.

    See also
    ========
    convert_equality_operators

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, convert_equals_signs)
    >>> parse_expr("1*2=x", transformations=(
    ... standard_transformations + (convert_equals_signs,)))
    Eq(2, x)
    >>> parse_expr("(1*2=x)=False", transformations=(
    ... standard_transformations + (convert_equals_signs,)))
    Eq(Eq(2, x), False)

    """
    res1 = _group_parentheses(convert_equals_signs)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _transform_equals_sign(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result