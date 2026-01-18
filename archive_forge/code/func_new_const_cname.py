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
def new_const_cname(self, prefix='', value=''):
    value = replace_identifier('_', value)[:32].strip('_')
    name_suffix = self.unique_const_cname(value + '{sep}{counter}')
    if prefix:
        prefix = Naming.interned_prefixes[prefix]
    else:
        prefix = Naming.const_prefix
    return '%s%s' % (prefix, name_suffix)