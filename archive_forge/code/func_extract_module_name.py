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
def extract_module_name(self, path, options):
    dir, filename = os.path.split(path)
    module_name, _ = os.path.splitext(filename)
    if '.' in module_name:
        return module_name
    names = [module_name]
    while self.is_package_dir(dir):
        parent, package_name = os.path.split(dir)
        if parent == dir:
            break
        names.append(package_name)
        dir = parent
    names.reverse()
    return '.'.join(names)