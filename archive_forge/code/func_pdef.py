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
@skip_doctest
@line_magic
def pdef(self, parameter_s='', namespaces=None):
    """Print the call signature for any callable object.

        If the object is a class, print the constructor information.

        Examples
        --------
        ::

          In [3]: %pdef urllib.urlopen
          urllib.urlopen(url, data=None, proxies=None)
        """
    self.shell._inspect('pdef', parameter_s, namespaces)