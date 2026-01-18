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
@contextmanager
def default_role(docname: str, name: str) -> Generator[None, None, None]:
    if name:
        dummy_reporter = Reporter('', 4, 4)
        role_fn, _ = roles.role(name, english, 0, dummy_reporter)
        if role_fn:
            docutils.register_role('', role_fn)
        else:
            logger.warning(__('default role %s not found'), name, location=docname)
    yield
    docutils.unregister_role('')