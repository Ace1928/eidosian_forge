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
@Utils.cached_function
def search_module_in_dir(package_dir, module_name, suffix):
    path = Utils.find_versioned_file(package_dir, module_name, suffix)
    if not path and suffix:
        path = Utils.find_versioned_file(os.path.join(package_dir, module_name), '__init__', suffix)
    return path