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
def lambda_notation(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Substitutes "lambda" with its SymPy equivalent Lambda().
    However, the conversion does not take place if only "lambda"
    is passed because that is a syntax error.

    """
    result: List[TOKEN] = []
    flag = False
    toknum, tokval = tokens[0]
    tokLen = len(tokens)
    if toknum == NAME and tokval == 'lambda':
        if tokLen == 2 or (tokLen == 3 and tokens[1][0] == NEWLINE):
            result.extend(tokens)
        elif tokLen > 2:
            result.extend([(NAME, 'Lambda'), (OP, '('), (OP, '('), (OP, ')'), (OP, ')')])
            for tokNum, tokVal in tokens[1:]:
                if tokNum == OP and tokVal == ':':
                    tokVal = ','
                    flag = True
                if not flag and tokNum == OP and (tokVal in ('*', '**')):
                    raise TokenError('Starred arguments in lambda not supported')
                if flag:
                    result.insert(-1, (tokNum, tokVal))
                else:
                    result.insert(-2, (tokNum, tokVal))
    else:
        result.extend(tokens)
    return result