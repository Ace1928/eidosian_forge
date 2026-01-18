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
def put_init_var_to_py_none(self, entry, template='%s', nanny=True):
    code = template % entry.cname
    self.put_init_to_py_none(code, entry.type, nanny)
    if entry.in_closure:
        self.put_giveref('Py_None')