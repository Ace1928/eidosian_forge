from __future__ import annotations
import logging  # isort:skip
import re
from typing import Any, Iterable, overload
from urllib.parse import quote_plus
def nice_join(seq: Iterable[str], *, sep: str=', ', conjunction: str='or') -> str:
    """ Join together sequences of strings into English-friendly phrases using
    the conjunction ``or`` when appropriate.

    Args:
        seq (seq[str]) : a sequence of strings to nicely join
        sep (str, optional) : a sequence delimiter to use (default: ", ")
        conjunction (str or None, optional) : a conjunction to use for the last
            two items, or None to reproduce basic join behaviour (default: "or")

    Returns:
        a joined string

    Examples:
        >>> nice_join(["a", "b", "c"])
        'a, b or c'

    """
    seq = [str(x) for x in seq]
    if len(seq) <= 1 or conjunction is None:
        return sep.join(seq)
    else:
        return f'{sep.join(seq[:-1])} {conjunction} {seq[-1]}'