from typing import Iterable, List, Tuple
from triad.constants import TRIAD_VAR_QUOTE
from .assertion import assert_or_throw
from .string import validate_triad_var_name
def safe_split_and_unquote(s: str, sep_char: str=',', quote: str=TRIAD_VAR_QUOTE, on_unquoted_empty: str='keep') -> List[str]:
    """Split the string and unquote every part

    .. admonition:: Examples

        ``" a , ` b ` , c "`` => ``["a", " b ","c"]``

    :param s: the original string
    :param sep_char: the split character, defaults to ","
    :param quote: the quote character
    :param on_unquoted_empty: can be ``keep``, ``ignore`` or
        ``throw``, defaults to "keep"
    :raises ValueError: if there are empty but unquoted parts and
        ``on_unquoted_empty`` is ``throw``
    :return: the unquoted parts.
    """
    res: List[str] = []
    for _s in safe_split_out_of_quote(s, sep_char, quote=quote):
        s = _s.strip()
        if s == '':
            if on_unquoted_empty == 'keep':
                res.append(s)
            elif on_unquoted_empty == 'ignore':
                continue
            else:
                raise ValueError(f'empty is not allowed in {s}')
        else:
            res.append(unquote_name(s, quote=quote))
    return res