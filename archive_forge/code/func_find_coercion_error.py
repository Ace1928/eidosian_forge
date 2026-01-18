from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
def find_coercion_error(type_tuple, default, env):
    err = coercion_error_dict.get(type_tuple)
    if err is None:
        return default
    elif env.directives['c_string_encoding'] and any((t in type_tuple for t in (PyrexTypes.c_char_ptr_type, PyrexTypes.c_uchar_ptr_type, PyrexTypes.c_const_char_ptr_type, PyrexTypes.c_const_uchar_ptr_type))):
        if type_tuple[1].is_pyobject:
            return default
        elif env.directives['c_string_encoding'] in ('ascii', 'default'):
            return default
        else:
            return "'%s' objects do not support coercion to C types with non-ascii or non-default c_string_encoding" % type_tuple[0].name
    else:
        return err