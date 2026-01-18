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
def get_int_const(self, str_value, longness=False):
    py_type = longness and 'long' or 'int'
    try:
        c = self.num_const_index[str_value, py_type]
    except KeyError:
        c = self.new_num_const(str_value, py_type)
    return c