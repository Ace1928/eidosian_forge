from typing import Iterable, List, Tuple
from triad.constants import TRIAD_VAR_QUOTE
from .assertion import assert_or_throw
from .string import validate_triad_var_name
def safe_replace_out_of_quote(s: str, find: str, replace: str, quote=TRIAD_VAR_QUOTE) -> str:
    """Replace strings out of the quoted part

    :param s: the original string
    :param find: the string to find
    :param replace: the string used to replace
    :param quote: the quote character

    :return: the string with the replacements
    """
    return ''.join((s[rg[1]:rg[2]] if rg[0] else s[rg[1]:rg[2]].replace(find, replace) for rg in split_quoted_string(s, quote=quote)))