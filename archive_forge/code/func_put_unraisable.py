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
def put_unraisable(self, qualified_name, nogil=False):
    """
        Generate code to print a Python warning for an unraisable exception.

        qualified_name should be the qualified name of the function.
        """
    format_tuple = (qualified_name, Naming.clineno_cname, Naming.lineno_cname, Naming.filename_cname, self.globalstate.directives['unraisable_tracebacks'], nogil)
    self.funcstate.uses_error_indicator = True
    self.putln('__Pyx_WriteUnraisable("%s", %s, %s, %s, %d, %d);' % format_tuple)
    self.globalstate.use_utility_code(UtilityCode.load_cached('WriteUnraisableException', 'Exceptions.c'))