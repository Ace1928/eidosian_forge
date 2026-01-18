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
def extract_symbols(code, symbols):
    """
    Return a tuple  (blocks, not_found)
    where ``blocks`` is a list of code fragments
    for each symbol parsed from code, and ``not_found`` are
    symbols not found in the code.

    For example::

        In [1]: code = '''a = 10
           ...: def b(): return 42
           ...: class A: pass'''

        In [2]: extract_symbols(code, 'A,b,z')
        Out[2]: (['class A: pass\\n', 'def b(): return 42\\n'], ['z'])
    """
    symbols = symbols.split(',')
    py_code = ast.parse(code)
    marks = [(getattr(s, 'name', None), s.lineno) for s in py_code.body]
    code = code.split('\n')
    symbols_lines = {}
    end = len(code)
    for name, start in reversed(marks):
        while not code[end - 1].strip():
            end -= 1
        if name:
            symbols_lines[name] = (start - 1, end)
        end = start - 1
    blocks = []
    not_found = []
    for symbol in symbols:
        if symbol in symbols_lines:
            start, end = symbols_lines[symbol]
            blocks.append('\n'.join(code[start:end]) + '\n')
        else:
            not_found.append(symbol)
    return (blocks, not_found)