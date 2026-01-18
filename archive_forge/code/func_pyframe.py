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
def pyframe(self, frame):
    pyframe = Frame(frame).get_pyop()
    if pyframe:
        return pyframe
    else:
        raise gdb.RuntimeError('Unable to find the Python frame, run your code with a debug build (configure with --with-pydebug or compile with -g).')