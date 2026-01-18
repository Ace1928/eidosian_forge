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
def put_var_declaration(self, entry, storage_class='', dll_linkage=None, definition=True):
    if entry.visibility == 'private' and (not (definition or entry.defined_in_pxd)):
        return
    if entry.visibility == 'private' and (not entry.used):
        return
    if not entry.cf_used:
        self.put('CYTHON_UNUSED ')
    if storage_class:
        self.put('%s ' % storage_class)
    if entry.is_cpp_optional:
        self.put(entry.type.cpp_optional_declaration_code(entry.cname, dll_linkage=dll_linkage))
    else:
        self.put(entry.type.declaration_code(entry.cname, dll_linkage=dll_linkage))
    if entry.init is not None:
        self.put_safe(' = %s' % entry.type.literal_code(entry.init))
    elif entry.type.is_pyobject:
        self.put(' = NULL')
    self.putln(';')
    self.funcstate.scope.use_entry_utility_code(entry)