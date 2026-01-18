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
def pinfo(self, parameter_s='', namespaces=None):
    """Provide detailed information about an object.

        '%pinfo object' is just a synonym for object? or ?object."""
    detail_level = 0
    pinfo, qmark1, oname, qmark2 = re.match('(pinfo )?(\\?*)(.*?)(\\??$)', parameter_s).groups()
    if pinfo or qmark1 or qmark2:
        detail_level = 1
    if '*' in oname:
        self.psearch(oname)
    else:
        self.shell._inspect('pinfo', oname, detail_level=detail_level, namespaces=namespaces)