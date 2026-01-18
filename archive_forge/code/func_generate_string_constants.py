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
def generate_string_constants(self):
    c_consts = [(len(c.cname), c.cname, c) for c in self.string_const_index.values()]
    c_consts.sort()
    py_strings = []
    decls_writer = self.parts['string_decls']
    for _, cname, c in c_consts:
        conditional = False
        if c.py_versions and (2 not in c.py_versions or 3 not in c.py_versions):
            conditional = True
            decls_writer.putln('#if PY_MAJOR_VERSION %s 3' % (2 in c.py_versions and '<' or '>='))
        decls_writer.putln('static const char %s[] = "%s";' % (cname, StringEncoding.split_string_literal(c.escaped_value)), safe=True)
        if conditional:
            decls_writer.putln('#endif')
        if c.py_strings is not None:
            for py_string in c.py_strings.values():
                py_strings.append((c.cname, len(py_string.cname), py_string))
    for c, cname in sorted(self.pyunicode_ptr_const_index.items()):
        utf16_array, utf32_array = StringEncoding.encode_pyunicode_string(c)
        if utf16_array:
            decls_writer.putln('#ifdef Py_UNICODE_WIDE')
        decls_writer.putln('static Py_UNICODE %s[] = { %s };' % (cname, utf32_array))
        if utf16_array:
            decls_writer.putln('#else')
            decls_writer.putln('static Py_UNICODE %s[] = { %s };' % (cname, utf16_array))
            decls_writer.putln('#endif')
    init_constants = self.parts['init_constants']
    if py_strings:
        self.use_utility_code(UtilityCode.load_cached('InitStrings', 'StringTools.c'))
        py_strings.sort()
        w = self.parts['pystring_table']
        w.putln('')
        w.putln('static int __Pyx_CreateStringTabAndInitStrings(void) {')
        w.putln('__Pyx_StringTabEntry %s[] = {' % Naming.stringtab_cname)
        for py_string_args in py_strings:
            c_cname, _, py_string = py_string_args
            if not py_string.is_str or not py_string.encoding or py_string.encoding in ('ASCII', 'USASCII', 'US-ASCII', 'UTF8', 'UTF-8'):
                encoding = '0'
            else:
                encoding = '"%s"' % py_string.encoding.lower()
            self.parts['module_state'].putln('PyObject *%s;' % py_string.cname)
            self.parts['module_state_defines'].putln('#define %s %s->%s' % (py_string.cname, Naming.modulestateglobal_cname, py_string.cname))
            self.parts['module_state_clear'].putln('Py_CLEAR(clear_module_state->%s);' % py_string.cname)
            self.parts['module_state_traverse'].putln('Py_VISIT(traverse_module_state->%s);' % py_string.cname)
            if py_string.py3str_cstring:
                w.putln('#if PY_MAJOR_VERSION >= 3')
                w.putln('{&%s, %s, sizeof(%s), %s, %d, %d, %d},' % (py_string.cname, py_string.py3str_cstring.cname, py_string.py3str_cstring.cname, '0', 1, 0, py_string.intern))
                w.putln('#else')
            w.putln('{&%s, %s, sizeof(%s), %s, %d, %d, %d},' % (py_string.cname, c_cname, c_cname, encoding, py_string.is_unicode, py_string.is_str, py_string.intern))
            if py_string.py3str_cstring:
                w.putln('#endif')
        w.putln('{0, 0, 0, 0, 0, 0, 0}')
        w.putln('};')
        w.putln('return __Pyx_InitStrings(%s);' % Naming.stringtab_cname)
        w.putln('}')
        init_constants.putln('if (__Pyx_CreateStringTabAndInitStrings() < 0) %s;' % init_constants.error_goto(self.module_pos))