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
def search_include_directories(self, qualified_name, suffix=None, source_pos=None, include=False, sys_path=False, source_file_path=None):
    include_dirs = self.include_directories
    if sys_path:
        include_dirs = include_dirs + sys.path
    include_dirs = tuple(include_dirs + [standard_include_path])
    return search_include_directories(include_dirs, qualified_name, suffix or '', source_pos, include, source_file_path)