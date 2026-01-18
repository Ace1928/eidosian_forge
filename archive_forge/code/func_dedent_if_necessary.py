from __future__ import absolute_import
import sys
import re
import itertools as _itertools
from codecs import BOM_UTF8
from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, \
from parso.python.token import PythonTokenTypes
from parso.utils import split_lines, PythonVersionInfo, parse_version_string
def dedent_if_necessary(start):
    while start < indents[-1]:
        if start > indents[-2]:
            yield PythonToken(ERROR_DEDENT, '', (lnum, start), '')
            indents[-1] = start
            break
        indents.pop()
        yield PythonToken(DEDENT, '', spos, '')