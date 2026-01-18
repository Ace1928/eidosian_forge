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
def simple_decorator(f):
    """
    A simple decorator that does nothing, for tests to use.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper