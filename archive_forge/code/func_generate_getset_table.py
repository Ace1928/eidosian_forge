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
def generate_getset_table(self, env, code):
    if env.property_entries:
        code.putln('')
        code.putln('static struct PyGetSetDef %s[] = {' % env.getset_table_cname)
        for entry in env.property_entries:
            doc = entry.doc
            if doc:
                if doc.is_unicode:
                    doc = doc.as_utf8_string()
                doc_code = 'PyDoc_STR(%s)' % doc.as_c_string_literal()
            else:
                doc_code = '0'
            code.putln('{(char *)%s, %s, %s, (char *)%s, 0},' % (entry.name.as_c_string_literal(), entry.getter_cname or '0', entry.setter_cname or '0', doc_code))
        code.putln('{0, 0, 0, 0, 0}')
        code.putln('};')