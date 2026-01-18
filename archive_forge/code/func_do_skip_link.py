import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def do_skip_link(text: str, index: Tuple[int, int]) -> bool:
    """Check if the identified URL is something other than a complete link.

    Is the identified link simply:
        1. The URL scheme pattern such as 's3://' or 'file://' or 'dns:'.
        2. The beginning of a URL link that has been wrapped by the user.

    Arguments
    ---------
    text : str
        The description text containing the link.
    index : tuple
        The index in the text of the starting and ending position of the
        identified link.

    Returns
    -------
    _do_skip : bool
        Whether to skip this link and simply treat it as a standard text word.
    """
    _do_skip = re.search(URL_SKIP_REGEX, text[index[0]:index[1]]) is not None
    with contextlib.suppress(IndexError):
        _do_skip = _do_skip or (text[index[0]] == '<' and text[index[1]] != '>')
    return _do_skip