import gc
import re
import sys
from IPython.core import page
from IPython.core.error import StdinNotImplementedError, UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.encoding import DEFAULT_ENCODING
from IPython.utils.openpy import read_py_file
from IPython.utils.path import get_py_filename
@line_magic
def pdoc(self, parameter_s='', namespaces=None):
    """Print the docstring for an object.

        If the given object is a class, it will print both the class and the
        constructor docstrings."""
    self.shell._inspect('pdoc', parameter_s, namespaces)