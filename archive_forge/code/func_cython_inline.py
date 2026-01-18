from __future__ import absolute_import, print_function
import io
import os
import re
import sys
import time
import copy
import distutils.log
import textwrap
import hashlib
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext
from IPython.core import display
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.utils.text import dedent
from ..Shadow import __version__ as cython_version
from ..Compiler.Errors import CompileError
from .Inline import cython_inline, load_dynamic
from .Dependencies import cythonize
from ..Utils import captured_fd, print_captured
@cell_magic
def cython_inline(self, line, cell):
    """Compile and run a Cython code cell using Cython.inline.

        This magic simply passes the body of the cell to Cython.inline
        and returns the result. If the variables `a` and `b` are defined
        in the user's namespace, here is a simple example that returns
        their sum::

            %%cython_inline
            return a+b

        For most purposes, we recommend the usage of the `%%cython` magic.
        """
    locs = self.shell.user_global_ns
    globs = self.shell.user_ns
    return cython_inline(cell, locals=locs, globals=globs)