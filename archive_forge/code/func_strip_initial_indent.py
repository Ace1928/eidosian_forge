import inspect
import io
import os
import re
import sys
import ast
from itertools import chain
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from pathlib import Path
from IPython.core.error import TryNext, StdinNotImplementedError, UsageError
from IPython.core.macro import Macro
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.oinspect import find_file, find_source_lines
from IPython.core.release import version
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import get_py_filename
from warnings import warn
from logging import error
from IPython.utils.text import get_text_list
def strip_initial_indent(lines):
    """For %load, strip indent from lines until finding an unindented line.

    https://github.com/ipython/ipython/issues/9775
    """
    indent_re = re.compile('\\s+')
    it = iter(lines)
    first_line = next(it)
    indent_match = indent_re.match(first_line)
    if indent_match:
        indent = indent_match.group()
        yield first_line[len(indent):]
        for line in it:
            if line.startswith(indent):
                yield line[len(indent):]
            else:
                yield line
                break
    else:
        yield first_line
    for line in it:
        yield line