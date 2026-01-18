import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
def parse_series(tokens: Iterable['Token']) -> Tuple[int, int]:
    """
    Parses the arguments for :nth-child() and friends.

    :raises: A list of tokens
    :returns: :``(a, b)``

    """
    for token in tokens:
        if token.type == 'STRING':
            raise ValueError('String tokens not allowed in series.')
    s = ''.join((typing.cast(str, token.value) for token in tokens)).strip()
    if s == 'odd':
        return (2, 1)
    elif s == 'even':
        return (2, 0)
    elif s == 'n':
        return (1, 0)
    if 'n' not in s:
        return (0, int(s))
    a, b = s.split('n', 1)
    a_as_int: int
    if not a:
        a_as_int = 1
    elif a == '-' or a == '+':
        a_as_int = int(a + '1')
    else:
        a_as_int = int(a)
    b_as_int: int
    if not b:
        b_as_int = 0
    else:
        b_as_int = int(b)
    return (a_as_int, b_as_int)