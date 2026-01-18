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
def initialize_main_c_code(self):
    rootwriter = self.rootwriter
    for i, part in enumerate(self.code_layout):
        w = self.parts[part] = rootwriter.insertion_point()
        if i > 0:
            w.putln('/* #### Code section: %s ### */' % part)
    if not Options.cache_builtins:
        del self.parts['cached_builtins']
    else:
        w = self.parts['cached_builtins']
        w.enter_cfunc_scope()
        w.putln('static CYTHON_SMALL_CODE int __Pyx_InitCachedBuiltins(void) {')
    w = self.parts['cached_constants']
    w.enter_cfunc_scope()
    w.putln('')
    w.putln('static CYTHON_SMALL_CODE int __Pyx_InitCachedConstants(void) {')
    w.put_declare_refcount_context()
    w.put_setup_refcount_context(StringEncoding.EncodedString('__Pyx_InitCachedConstants'))
    w = self.parts['init_globals']
    w.enter_cfunc_scope()
    w.putln('')
    w.putln('static CYTHON_SMALL_CODE int __Pyx_InitGlobals(void) {')
    w = self.parts['init_constants']
    w.enter_cfunc_scope()
    w.putln('')
    w.putln('static CYTHON_SMALL_CODE int __Pyx_InitConstants(void) {')
    if not Options.generate_cleanup_code:
        del self.parts['cleanup_globals']
    else:
        w = self.parts['cleanup_globals']
        w.enter_cfunc_scope()
        w.putln('')
        w.putln('static CYTHON_SMALL_CODE void __Pyx_CleanupGlobals(void) {')
    code = self.parts['utility_code_proto']
    code.putln('')
    code.putln('/* --- Runtime support code (head) --- */')
    code = self.parts['utility_code_def']
    if self.code_config.emit_linenums:
        code.write('\n#line 1 "cython_utility"\n')
    code.putln('')
    code.putln('/* --- Runtime support code --- */')