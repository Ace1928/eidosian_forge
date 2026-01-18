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
def put_pymethoddef(self, entry, term, allow_skip=True, wrapper_code_writer=None):
    is_reverse_number_slot = False
    if entry.is_special or entry.name == '__getattribute__':
        from . import TypeSlots
        is_reverse_number_slot = True
        if entry.name not in special_py_methods and (not TypeSlots.is_reverse_number_slot(entry.name)):
            if entry.name == '__getattr__' and (not self.globalstate.directives['fast_getattr']):
                pass
            elif allow_skip:
                return
    method_flags = entry.signature.method_flags()
    if not method_flags:
        return
    if entry.is_special:
        method_flags += [TypeSlots.method_coexist]
    func_ptr = wrapper_code_writer.put_pymethoddef_wrapper(entry) if wrapper_code_writer else entry.func_cname
    cast = entry.signature.method_function_type()
    if cast != 'PyCFunction':
        func_ptr = '(void*)(%s)%s' % (cast, func_ptr)
    entry_name = entry.name.as_c_string_literal()
    if is_reverse_number_slot:
        slot = TypeSlots.get_slot_table(self.globalstate.directives).get_slot_by_method_name(entry.name)
        preproc_guard = slot.preprocessor_guard_code()
        if preproc_guard:
            self.putln(preproc_guard)
    self.putln('{%s, (PyCFunction)%s, %s, %s}%s' % (entry_name, func_ptr, '|'.join(method_flags), entry.doc_cname if entry.doc else '0', term))
    if is_reverse_number_slot and preproc_guard:
        self.putln('#endif')