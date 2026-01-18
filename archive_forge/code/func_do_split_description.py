import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def do_split_description(text: str, indentation: str, wrap_length: int, style: str) -> Union[List[str], Iterable]:
    """Split the description into a list of lines.

    Parameters
    ----------
    text : str
        The docstring description.
    indentation : str
        The indentation (number of spaces or tabs) to place in front of each
        line.
    wrap_length : int
        The column to wrap each line at.
    style : str
        The docstring style to use for dealing with parameter lists.

    Returns
    -------
    _lines : list
        A list containing each line of the description with any links put
        back together.
    """
    _lines: List[str] = []
    _text_idx = 0
    _url_idx = do_find_links(text)
    _field_idx, _wrap_fields = do_find_field_lists(text, style)
    _url_idx = _field_over_url(_field_idx, _url_idx)
    if not _url_idx and (not (_field_idx and _wrap_fields)):
        return description_to_list(text, indentation, wrap_length)
    if _url_idx:
        _lines, _text_idx = do_wrap_urls(text, _url_idx, 0, indentation, wrap_length)
    if _field_idx:
        _lines, _text_idx = do_wrap_field_lists(text, _field_idx, _lines, _text_idx, indentation, wrap_length)
    else:
        _lines += _do_close_description(text, _text_idx, indentation)
    return _lines