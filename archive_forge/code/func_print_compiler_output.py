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
def print_compiler_output(stdout, stderr, where):
    print_captured(stdout, where, u'Content of stdout:\n')
    print_captured(stderr, where, u'Content of stderr:\n')