import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def do_clean_url(url: str, indentation: str) -> str:
    """Strip newlines and multiple whitespace from URL string.

    This function deals with situations such as:

    `Get\\n     Cookies.txt <https://chrome.google.com/webstore/detail/get-

    by returning:
    `Get Cookies.txt <https://chrome.google.com/webstore/detail/get-

    Parameters
    ----------
    url : str
        The URL that was found by the do_find_links() function and needs to be
        processed.
    indentation : str
       The indentation pattern used.
    Returns
    -------
    url : str
       The URL with internal newlines removed and excess whitespace removed.
    """
    _lines = url.splitlines()
    for _idx, _line in enumerate(_lines):
        if indentation and _line[:len(indentation)] == indentation:
            _lines[_idx] = f' {_line.strip()}'
    return f'{indentation}{''.join(list(_lines))}'