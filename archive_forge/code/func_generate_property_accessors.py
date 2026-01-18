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
def generate_property_accessors(self, cclass_scope, code):
    for entry in cclass_scope.property_entries:
        property_scope = entry.scope
        if property_scope.defines_any(['__get__']):
            self.generate_property_get_function(entry, code)
        if property_scope.defines_any(['__set__', '__del__']):
            self.generate_property_set_function(entry, code)