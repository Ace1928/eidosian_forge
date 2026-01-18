from __future__ import absolute_import, print_function
import os
import re
import sys
import io
from . import Errors
from .StringEncoding import EncodedString
from .Scanning import PyrexScanner, FileSourceDescriptor
from .Errors import PyrexError, CompileError, error, warning
from .Symtab import ModuleScope
from .. import Utils
from . import Options
from .Options import CompilationOptions, default_options
from .CmdLine import parse_command_line
from .Lexicon import (unicode_start_ch_any, unicode_continuation_ch_any,
def c_file_out_of_date(self, source_path, output_path):
    if not os.path.exists(output_path):
        return 1
    c_time = Utils.modification_time(output_path)
    if Utils.file_newer_than(source_path, c_time):
        return 1
    pxd_path = Utils.replace_suffix(source_path, '.pxd')
    if os.path.exists(pxd_path) and Utils.file_newer_than(pxd_path, c_time):
        return 1
    for kind, name in self.read_dependency_file(source_path):
        if kind == 'cimport':
            dep_path = self.find_pxd_file(name, source_file_path=source_path)
        elif kind == 'include':
            dep_path = self.search_include_directories(name, source_file_path=source_path)
        else:
            continue
        if dep_path and Utils.file_newer_than(dep_path, c_time):
            return 1
    return 0