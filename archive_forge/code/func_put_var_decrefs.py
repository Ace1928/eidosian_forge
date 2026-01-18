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
def put_var_decrefs(self, entries, used_only=0):
    for entry in entries:
        if not used_only or entry.used:
            if entry.xdecref_cleanup:
                self.put_var_xdecref(entry)
            else:
                self.put_var_decref(entry)