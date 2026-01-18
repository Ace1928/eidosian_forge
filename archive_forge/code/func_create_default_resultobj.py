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
def create_default_resultobj(compilation_source, options):
    result = CompilationResult()
    result.main_source_file = compilation_source.source_desc.filename
    result.compilation_source = compilation_source
    source_desc = compilation_source.source_desc
    result.c_file = get_output_filename(source_desc.filename, compilation_source.cwd, options)
    result.embedded_metadata = options.embedded_metadata
    return result