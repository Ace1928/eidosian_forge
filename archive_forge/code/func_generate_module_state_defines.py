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
def generate_module_state_defines(self, env, code):
    code.putln('#define %s %s->%s' % (env.module_dict_cname, Naming.modulestateglobal_cname, env.module_dict_cname))
    code.putln('#define %s %s->%s' % (Naming.builtins_cname, Naming.modulestateglobal_cname, Naming.builtins_cname))
    code.putln('#define %s %s->%s' % (Naming.cython_runtime_cname, Naming.modulestateglobal_cname, Naming.cython_runtime_cname))
    code.putln('#define %s %s->%s' % (Naming.empty_tuple, Naming.modulestateglobal_cname, Naming.empty_tuple))
    code.putln('#define %s %s->%s' % (Naming.empty_bytes, Naming.modulestateglobal_cname, Naming.empty_bytes))
    code.putln('#define %s %s->%s' % (Naming.empty_unicode, Naming.modulestateglobal_cname, Naming.empty_unicode))
    if Options.pre_import is not None:
        code.putln('#define %s %s->%s' % (Naming.preimport_cname, Naming.modulestateglobal_cname, Naming.preimport_cname))
    for cname, used_name in Naming.used_types_and_macros:
        code.putln('#ifdef %s' % used_name)
        code.putln('#define %s %s->%s' % (cname, Naming.modulestateglobal_cname, cname))
        code.putln('#endif')