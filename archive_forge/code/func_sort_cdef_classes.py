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
def sort_cdef_classes(self, env):
    key_func = operator.attrgetter('objstruct_cname')
    entry_dict, entry_order = ({}, [])
    for entry in env.c_class_entries:
        key = key_func(entry.type)
        assert key not in entry_dict, key
        entry_dict[key] = entry
        entry_order.append(key)
    env.c_class_entries[:] = self.sort_types_by_inheritance(entry_dict, entry_order, key_func)