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
def compile_multiple(sources, options):
    """
    compile_multiple(sources, options)

    Compiles the given sequence of Pyrex implementation files and returns
    a CompilationResultSet. Performs timestamp checking and/or recursion
    if these are specified in the options.
    """
    if len(sources) > 1 and options.module_name:
        raise RuntimeError('Full module name can only be set for single source compilation')
    sources = [os.path.abspath(source) for source in sources]
    processed = set()
    results = CompilationResultSet()
    timestamps = options.timestamps
    verbose = options.verbose
    context = None
    cwd = os.getcwd()
    for source in sources:
        if source not in processed:
            if context is None:
                context = Context.from_options(options)
            output_filename = get_output_filename(source, cwd, options)
            out_of_date = context.c_file_out_of_date(source, output_filename)
            if not timestamps or out_of_date:
                if verbose:
                    sys.stderr.write('Compiling %s\n' % source)
                result = run_pipeline(source, options, full_module_name=options.module_name, context=context)
                results.add(source, result)
                context = None
            processed.add(source)
    return results