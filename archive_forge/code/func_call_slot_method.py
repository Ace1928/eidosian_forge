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
def call_slot_method(method_name, reverse):
    func_cname = get_slot_method_cname(method_name)
    if func_cname:
        return '%s(%s%s)' % (func_cname, 'right, left' if reverse else 'left, right', extra_arg)
    else:
        return '%s_maybe_call_slot(__Pyx_PyType_GetSlot(%s, tp_base, PyTypeObject*), left, right %s)' % (func_name, scope.parent_type.typeptr_cname, extra_arg)