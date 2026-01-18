from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def readcode(self, expr):
    if expr:
        return (expr, PythonCodeExecutor.Py_single_input)
    else:
        lines = []
        while True:
            try:
                if sys.version_info[0] == 2:
                    line = raw_input()
                else:
                    line = input('>')
            except EOFError:
                break
            else:
                if line.rstrip() == 'end':
                    break
                lines.append(line)
        return ('\n'.join(lines), PythonCodeExecutor.Py_file_input)