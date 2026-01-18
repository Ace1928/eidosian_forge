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
def auto_number(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """
    Converts numeric literals to use SymPy equivalents.

    Complex numbers use ``I``, integer literals use ``Integer``, and float
    literals use ``Float``.

    """
    result: List[TOKEN] = []
    for toknum, tokval in tokens:
        if toknum == NUMBER:
            number = tokval
            postfix = []
            if number.endswith('j') or number.endswith('J'):
                number = number[:-1]
                postfix = [(OP, '*'), (NAME, 'I')]
            if '.' in number or (('e' in number or 'E' in number) and (not (number.startswith('0x') or number.startswith('0X')))):
                seq = [(NAME, 'Float'), (OP, '('), (NUMBER, repr(str(number))), (OP, ')')]
            else:
                seq = [(NAME, 'Integer'), (OP, '('), (NUMBER, number), (OP, ')')]
            result.extend(seq + postfix)
        else:
            result.append((toknum, tokval))
    return result