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
def process_pxd(self, source_desc, scope, module_name):
    from . import Pipeline
    if isinstance(source_desc, FileSourceDescriptor) and source_desc._file_type == 'pyx':
        source = CompilationSource(source_desc, module_name, os.getcwd())
        result_sink = create_default_resultobj(source, self.options)
        pipeline = Pipeline.create_pyx_as_pxd_pipeline(self, result_sink)
        result = Pipeline.run_pipeline(pipeline, source)
    else:
        pipeline = Pipeline.create_pxd_pipeline(self, scope, module_name)
        result = Pipeline.run_pipeline(pipeline, source_desc)
    return result