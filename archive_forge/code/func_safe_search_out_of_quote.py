from typing import Iterable, List, Tuple
from triad.constants import TRIAD_VAR_QUOTE
from .assertion import assert_or_throw
from .string import validate_triad_var_name
def safe_search_out_of_quote(s: str, chars: str, quote=TRIAD_VAR_QUOTE) -> Iterable[Tuple[int, str]]:
    """Search for chars out of the quoted parts

    :param s: the original string
    :param chars: the charaters to find
    :param quote: the quote character
    :yield: the tuple in format of ``position, char``
    """
    for rg in split_quoted_string(s, quote=quote):
        if not rg[0]:
            for i in range(rg[1], rg[2]):
                if s[i] in chars:
                    yield (i, s[i])