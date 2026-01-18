from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def from_html_one(html_code, **kwargs):
    """
    Generates a PrettyTables from a string of HTML code which contains only a
    single <table>
    """
    tables = from_html(html_code, **kwargs)
    try:
        assert len(tables) == 1
    except AssertionError:
        msg = 'More than one <table> in provided HTML code. Use from_html instead.'
        raise ValueError(msg)
    return tables[0]