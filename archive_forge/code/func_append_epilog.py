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
def append_epilog(content: StringList, epilog: str) -> None:
    """Append a string to content body as epilog."""
    if epilog:
        if 0 < len(content):
            source, lineno = content.info(-1)
        else:
            source = '<generated>'
            lineno = 0
        content.append('', source, lineno + 1)
        for lineno, line in enumerate(epilog.splitlines()):
            content.append(line, '<rst_epilog>', lineno)