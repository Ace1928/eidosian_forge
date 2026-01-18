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
def pfile(self, parameter_s='', namespaces=None):
    """Print (or run through pager) the file where an object is defined.

        The file opens at the line where the object definition begins. IPython
        will honor the environment variable PAGER if set, and otherwise will
        do its best to print the file in a convenient form.

        If the given argument is not an object currently defined, IPython will
        try to interpret it as a filename (automatically adding a .py extension
        if needed). You can thus use %pfile as a syntax highlighting code
        viewer."""
    out = self.shell._inspect('pfile', parameter_s, namespaces)
    if out == 'not found':
        try:
            filename = get_py_filename(parameter_s)
        except IOError as msg:
            print(msg)
            return
        page.page(self.shell.pycolorize(read_py_file(filename, skip_encoding_cookie=False)))