from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def from_html(html_code, **kwargs):
    """
    Generates a list of PrettyTables from a string of HTML code. Each <table> in
    the HTML becomes one PrettyTable object.
    """
    parser = TableHandler(**kwargs)
    parser.feed(html_code)
    return parser.tables