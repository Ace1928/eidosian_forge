import inspect
import itertools
import re
import tokenize
from collections import OrderedDict
from inspect import Signature
from token import DEDENT, INDENT, NAME, NEWLINE, NUMBER, OP, STRING
from tokenize import COMMENT, NL
from typing import Any, Dict, List, Optional, Tuple
from sphinx.pycode.ast import ast  # for py37 or older
from sphinx.pycode.ast import parse, unparse
def dedent_docstring(s: str) -> str:
    """Remove common leading indentation from docstring."""

    def dummy() -> None:
        pass
    dummy.__doc__ = s
    docstring = inspect.getdoc(dummy)
    if docstring:
        return docstring.lstrip('\r\n').rstrip('\r\n')
    else:
        return ''