import difflib
import os
import sys
import textwrap
from typing import Any, Optional, Tuple, Union
def format_repr(obj: Any, max_len: int=50, ellipsis: str='...') -> str:
    """Wrapper around `repr()` to print shortened and formatted string version.

    obj: The object to represent.
    max_len (int): Maximum string length. Longer strings will be cut in the
        middle so only the beginning and end is displayed, separated by ellipsis.
    ellipsis (str): Ellipsis character(s), e.g. "...".
    RETURNS (str): The formatted representation.
    """
    string = repr(obj)
    if len(string) >= max_len:
        half = int(max_len / 2)
        return '{} {} {}'.format(string[:half], ellipsis, string[-half:])
    else:
        return string