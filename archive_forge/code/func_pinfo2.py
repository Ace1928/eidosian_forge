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
def pinfo2(self, parameter_s='', namespaces=None):
    """Provide extra detailed information about an object.

        '%pinfo2 object' is just a synonym for object?? or ??object."""
    self.shell._inspect('pinfo', parameter_s, detail_level=1, namespaces=namespaces)