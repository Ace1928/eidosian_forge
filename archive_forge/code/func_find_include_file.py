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
def find_include_file(self, filename, pos=None, source_file_path=None):
    path = self.search_include_directories(filename, source_pos=pos, include=True, source_file_path=source_file_path)
    if not path:
        error(pos, "'%s' not found" % filename)
    return path