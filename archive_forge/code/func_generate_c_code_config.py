from __future__ import absolute_import
import cython
from collections import defaultdict
import json
import operator
import os
import re
import sys
from .PyrexTypes import CPtrType
from . import Future
from . import Annotate
from . import Code
from . import Naming
from . import Nodes
from . import Options
from . import TypeSlots
from . import PyrexTypes
from . import Pythran
from .Errors import error, warning, CompileError
from .PyrexTypes import py_object_type
from ..Utils import open_new_file, replace_suffix, decode_filename, build_hex_version, is_cython_generated_file
from .Code import UtilityCode, IncludeCode, TempitaUtilityCode
from .StringEncoding import EncodedString, encoded_string_or_bytes_literal
from .Pythran import has_np_pythran
def generate_c_code_config(env, options):
    if Options.annotate or options.annotate:
        emit_linenums = False
    else:
        emit_linenums = options.emit_linenums
    if hasattr(options, 'emit_code_comments'):
        print('Warning: option emit_code_comments is deprecated. Instead, use compiler directive emit_code_comments.')
    return Code.CCodeConfig(emit_linenums=emit_linenums, emit_code_comments=env.directives['emit_code_comments'], c_line_in_traceback=options.c_line_in_traceback)