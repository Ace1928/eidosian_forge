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
def cached_unbound_method_call_code(self, obj_cname, type_cname, method_name, arg_cnames):
    utility_code_name = 'CallUnboundCMethod%d' % len(arg_cnames)
    self.use_utility_code(UtilityCode.load_cached(utility_code_name, 'ObjectHandling.c'))
    cache_cname = self.get_cached_unbound_method(type_cname, method_name)
    args = [obj_cname] + arg_cnames
    return '__Pyx_%s(&%s, %s)' % (utility_code_name, cache_cname, ', '.join(args))