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
def generate_includes(self, env, cimported_modules, code, early=True, late=True):
    for inc in sorted(env.c_includes.values(), key=IncludeCode.sortkey):
        if inc.location == inc.EARLY:
            if early:
                inc.write(code)
        elif inc.location == inc.LATE:
            if late:
                inc.write(code)
    if early:
        code.putln_openmp('#include <omp.h>')