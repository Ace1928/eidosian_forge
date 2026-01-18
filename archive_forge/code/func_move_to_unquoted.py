from typing import Iterable, List, Tuple
from triad.constants import TRIAD_VAR_QUOTE
from .assertion import assert_or_throw
from .string import validate_triad_var_name
def move_to_unquoted(expr: str, p: int, quote=TRIAD_VAR_QUOTE) -> int:
    """When ``p`` is on a quote, find the position next to the end of
    the quoted part

    :param expr: the original string
    :param p: the current position of ``expr``, and it should be a quote
    :param quote: the quote character
    :raises SyntaxError: if there is an open quote detected
    :return: the position next to the end of the quoted part
    """
    e = p + 1
    le = len(expr)
    while e < le:
        if expr[e] == quote:
            if e + 1 < le and expr[e + 1] == quote:
                e = e + 2
            else:
                return e + 1
        else:
            e += 1
    raise SyntaxError(f'{expr} contains open quote {quote}')