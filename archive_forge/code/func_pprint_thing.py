from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
def pprint_thing(thing: Any, _nest_lvl: int=0, escape_chars: EscapeChars | None=None, default_escapes: bool=False, quote_strings: bool=False, max_seq_items: int | None=None) -> str:
    """
    This function is the sanctioned way of converting objects
    to a string representation and properly handles nested sequences.

    Parameters
    ----------
    thing : anything to be formatted
    _nest_lvl : internal use only. pprint_thing() is mutually-recursive
        with pprint_sequence, this argument is used to keep track of the
        current nesting level, and limit it.
    escape_chars : list or dict, optional
        Characters to escape. If a dict is passed the values are the
        replacements
    default_escapes : bool, default False
        Whether the input escape characters replaces or adds to the defaults
    max_seq_items : int or None, default None
        Pass through to other pretty printers to limit sequence printing

    Returns
    -------
    str
    """

    def as_escaped_string(thing: Any, escape_chars: EscapeChars | None=escape_chars) -> str:
        translate = {'\t': '\\t', '\n': '\\n', '\r': '\\r'}
        if isinstance(escape_chars, dict):
            if default_escapes:
                translate.update(escape_chars)
            else:
                translate = escape_chars
            escape_chars = list(escape_chars.keys())
        else:
            escape_chars = escape_chars or ()
        result = str(thing)
        for c in escape_chars:
            result = result.replace(c, translate[c])
        return result
    if hasattr(thing, '__next__'):
        return str(thing)
    elif isinstance(thing, dict) and _nest_lvl < get_option('display.pprint_nest_depth'):
        result = _pprint_dict(thing, _nest_lvl, quote_strings=True, max_seq_items=max_seq_items)
    elif is_sequence(thing) and _nest_lvl < get_option('display.pprint_nest_depth'):
        result = _pprint_seq(thing, _nest_lvl, escape_chars=escape_chars, quote_strings=quote_strings, max_seq_items=max_seq_items)
    elif isinstance(thing, str) and quote_strings:
        result = f"'{as_escaped_string(thing)}'"
    else:
        result = as_escaped_string(thing)
    return result