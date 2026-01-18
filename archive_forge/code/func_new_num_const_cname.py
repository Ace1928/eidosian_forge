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
def new_num_const_cname(self, value, py_type):
    if py_type == 'long':
        value += 'L'
        py_type = 'int'
    prefix = Naming.interned_prefixes[py_type]
    value = value.replace('.', '_').replace('+', '_').replace('-', 'neg_')
    if len(value) > 42:
        cname = self.unique_const_cname(prefix + 'large{counter}_' + value[:18] + '_xxx_' + value[-18:])
    else:
        cname = '%s%s' % (prefix, value)
    return cname