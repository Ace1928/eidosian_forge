import functools
import os
import re
import sys
import warnings
from io import StringIO
from typing import IO, Any, Dict, Generator, List, Optional, Pattern
from xml.etree import ElementTree
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives, roles
from sphinx import application, locale
from sphinx.pycode import ModuleAnalyzer
from sphinx.testing.path import path
from sphinx.util.osutil import relpath
def assert_not_re_search(regex: Pattern, text: str, flags: int=0) -> None:
    if re.search(regex, text, flags):
        raise AssertionError('%r did match %r' % (regex, text))