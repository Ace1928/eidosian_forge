from typing import Iterable, List, Tuple
from triad.constants import TRIAD_VAR_QUOTE
from .assertion import assert_or_throw
from .string import validate_triad_var_name
def split_quoted_string(s: str, quote=TRIAD_VAR_QUOTE) -> Iterable[Tuple[bool, int, int]]:
    """Split ``s`` to a sequence of quoted and unquoted parts.

    :param s: the original string
    :param quote: the quote character

    :yield: the tuple in the format of ``is_quoted, start, end``
    """
    b, e = (0, 0)
    le = len(s)
    while e < le:
        if s[e] == quote:
            if e > b:
                yield (False, b, e)
            b = e
            e = move_to_unquoted(s, e, quote=quote)
            yield (True, b, e)
            b = e
        else:
            e += 1
    if b < le:
        yield (False, b, le)