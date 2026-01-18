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
def entry_as_pyobject(self, entry):
    type = entry.type
    if not entry.is_self_arg and (not entry.type.is_complete()) or entry.type.is_extension_type:
        return '(PyObject *)' + entry.cname
    else:
        return entry.cname