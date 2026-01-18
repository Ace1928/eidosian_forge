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
def find_pyx_file(self, qualified_name, pos=None, sys_path=True, source_file_path=None):
    return self.search_include_directories(qualified_name, suffix='.pyx', source_pos=pos, sys_path=sys_path, source_file_path=source_file_path)