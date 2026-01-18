from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
def put_add_traceback(self, qualified_name, include_cline=True):
    """
        Build a Python traceback for propagating exceptions.

        qualified_name should be the qualified name of the function.
        """
    qualified_name = qualified_name.as_c_string_literal()
    format_tuple = (qualified_name, Naming.clineno_cname if include_cline else 0, Naming.lineno_cname, Naming.filename_cname)
    self.funcstate.uses_error_indicator = True
    self.putln('__Pyx_AddTraceback(%s, %s, %s, %s);' % format_tuple)