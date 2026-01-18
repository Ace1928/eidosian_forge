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
def put_release_gil(self, variable=None, unknown_gil_state=True):
    """Release the GIL, corresponds to `put_acquire_gil`."""
    self.use_fast_gil_utility_code()
    self.putln('#ifdef WITH_THREAD')
    self.putln('PyThreadState *_save;')
    self.putln('_save = NULL;')
    if unknown_gil_state:
        self.putln('if (PyGILState_Check()) {')
    self.putln('Py_UNBLOCK_THREADS')
    if unknown_gil_state:
        self.putln('}')
    if variable:
        self.putln('%s = _save;' % variable)
    self.putln('__Pyx_FastGIL_Remember();')
    self.putln('#endif')