from __future__ import absolute_import, print_function
import cython
from .. import __version__
import collections
import contextlib
import hashlib
import os
import shutil
import subprocess
import re, sys, time
from glob import iglob
from io import open as io_open
from os.path import relpath as _relpath
import zipfile
from .. import Utils
from ..Utils import (cached_function, cached_method, path_exists,
from ..Compiler import Errors
from ..Compiler.Main import Context
from ..Compiler.Options import (CompilationOptions, default_options,
def fix_windows_unicode_modules(module_list):
    if sys.platform != 'win32':
        return
    if sys.version_info < (3, 5) or sys.version_info >= (3, 8, 2):
        return

    def make_filtered_list(ignored_symbol, old_entries):

        class FilteredExportSymbols(list):

            def __contains__(self, val):
                return val == ignored_symbol or list.__contains__(self, val)
        filtered_list = FilteredExportSymbols(old_entries)
        if old_entries:
            filtered_list.extend((name for name in old_entries if name != ignored_symbol))
        return filtered_list
    for m in module_list:
        try:
            m.name.encode('ascii')
            continue
        except UnicodeEncodeError:
            pass
        m.export_symbols = make_filtered_list('PyInit_' + m.name.rsplit('.', 1)[-1], m.export_symbols)