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
def cython_pyximport(self, line, cell):
    """Compile and import a Cython code cell using pyximport.

        The contents of the cell are written to a `.pyx` file in the current
        working directory, which is then imported using `pyximport`. This
        magic requires a module name to be passed::

            %%cython_pyximport modulename
            def f(x):
                return 2.0*x

        The compiled module is then imported and all of its symbols are
        injected into the user's namespace. For most purposes, we recommend
        the usage of the `%%cython` magic.
        """
    module_name = line.strip()
    if not module_name:
        raise ValueError('module name must be given')
    fname = module_name + '.pyx'
    with io.open(fname, 'w', encoding='utf-8') as f:
        f.write(cell)
    if 'pyximport' not in sys.modules or not self._pyximport_installed:
        import pyximport
        pyximport.install()
        self._pyximport_installed = True
    if module_name in self._reloads:
        module = self._reloads[module_name]
    else:
        __import__(module_name)
        module = sys.modules[module_name]
        self._reloads[module_name] = module
    self._import_all(module)