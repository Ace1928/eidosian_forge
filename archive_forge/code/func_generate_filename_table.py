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
def generate_filename_table(self, code):
    from os.path import isabs, basename
    code.putln('')
    code.putln('static const char *%s[] = {' % Naming.filetable_cname)
    if code.globalstate.filename_list:
        for source_desc in code.globalstate.filename_list:
            file_path = source_desc.get_filenametable_entry()
            if isabs(file_path):
                file_path = basename(file_path)
            escaped_filename = file_path.replace('\\', '\\\\').replace('"', '\\"')
            escaped_filename = as_encoded_filename(escaped_filename)
            code.putln('%s,' % escaped_filename.as_c_string_literal())
    else:
        code.putln('0')
    code.putln('};')