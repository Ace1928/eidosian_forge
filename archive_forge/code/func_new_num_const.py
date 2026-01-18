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
def new_num_const(self, value, py_type, value_code=None):
    cname = self.new_num_const_cname(value, py_type)
    c = NumConst(cname, value, py_type, value_code)
    self.num_const_index[value, py_type] = c
    return c