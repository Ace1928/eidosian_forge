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
def put_error_if_unbound(self, pos, entry, in_nogil_context=False, unbound_check_code=None):
    if entry.from_closure:
        func = '__Pyx_RaiseClosureNameError'
        self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseClosureNameError', 'ObjectHandling.c'))
    elif entry.type.is_memoryviewslice and in_nogil_context:
        func = '__Pyx_RaiseUnboundMemoryviewSliceNogil'
        self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseUnboundMemoryviewSliceNogil', 'ObjectHandling.c'))
    elif entry.type.is_cpp_class and entry.is_cglobal:
        func = '__Pyx_RaiseCppGlobalNameError'
        self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseCppGlobalNameError', 'ObjectHandling.c'))
    elif entry.type.is_cpp_class and entry.is_variable and (not entry.is_member) and entry.scope.is_c_class_scope:
        func = '__Pyx_RaiseCppAttributeError'
        self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseCppAttributeError', 'ObjectHandling.c'))
    else:
        func = '__Pyx_RaiseUnboundLocalError'
        self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseUnboundLocalError', 'ObjectHandling.c'))
    if not unbound_check_code:
        unbound_check_code = entry.type.check_for_null_code(entry.cname)
    self.putln('if (unlikely(!%s)) { %s("%s"); %s }' % (unbound_check_code, func, entry.name, self.error_goto(pos)))