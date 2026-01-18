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
def split_symbols_custom(predicate: Callable[[str], bool]):
    """Creates a transformation that splits symbol names.

    ``predicate`` should return True if the symbol name is to be split.

    For instance, to retain the default behavior but avoid splitting certain
    symbol names, a predicate like this would work:


    >>> from sympy.parsing.sympy_parser import (parse_expr, _token_splittable,
    ... standard_transformations, implicit_multiplication,
    ... split_symbols_custom)
    >>> def can_split(symbol):
    ...     if symbol not in ('list', 'of', 'unsplittable', 'names'):
    ...             return _token_splittable(symbol)
    ...     return False
    ...
    >>> transformation = split_symbols_custom(can_split)
    >>> parse_expr('unsplittable', transformations=standard_transformations +
    ... (transformation, implicit_multiplication))
    unsplittable
    """

    def _split_symbols(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
        result: List[TOKEN] = []
        split = False
        split_previous = False
        for tok in tokens:
            if split_previous:
                split_previous = False
                continue
            split_previous = False
            if tok[0] == NAME and tok[1] in ['Symbol', 'Function']:
                split = True
            elif split and tok[0] == NAME:
                symbol = tok[1][1:-1]
                if predicate(symbol):
                    tok_type = result[-2][1]
                    del result[-2:]
                    i = 0
                    while i < len(symbol):
                        char = symbol[i]
                        if char in local_dict or char in global_dict:
                            result.append((NAME, '%s' % char))
                        elif char.isdigit():
                            chars = [char]
                            for i in range(i + 1, len(symbol)):
                                if not symbol[i].isdigit():
                                    i -= 1
                                    break
                                chars.append(symbol[i])
                            char = ''.join(chars)
                            result.extend([(NAME, 'Number'), (OP, '('), (NAME, "'%s'" % char), (OP, ')')])
                        else:
                            use = tok_type if i == len(symbol) else 'Symbol'
                            result.extend([(NAME, use), (OP, '('), (NAME, "'%s'" % char), (OP, ')')])
                        i += 1
                    split = False
                    split_previous = True
                    continue
                else:
                    split = False
            result.append(tok)
        return result
    return _split_symbols