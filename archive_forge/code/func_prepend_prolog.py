import re
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Generator
from unicodedata import east_asian_width
from docutils.parsers.rst import roles
from docutils.parsers.rst.languages import en as english
from docutils.statemachine import StringList
from docutils.utils import Reporter
from jinja2 import Environment
from sphinx.locale import __
from sphinx.util import docutils, logging
def prepend_prolog(content: StringList, prolog: str) -> None:
    """Prepend a string to content body as prolog."""
    if prolog:
        pos = 0
        for line in content:
            if docinfo_re.match(line):
                pos += 1
            else:
                break
        if pos > 0:
            content.insert(pos, '', '<generated>', 0)
            pos += 1
        for lineno, line in enumerate(prolog.splitlines()):
            content.insert(pos + lineno, line, '<rst_prolog>', lineno)
        content.insert(pos + lineno + 1, '', '<generated>', 0)