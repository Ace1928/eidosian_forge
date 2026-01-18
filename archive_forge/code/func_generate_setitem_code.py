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
def generate_setitem_code(self, value_code, code):
    if self.index.type.is_int:
        if self.base.type is bytearray_type:
            code.globalstate.use_utility_code(UtilityCode.load_cached('SetItemIntByteArray', 'StringTools.c'))
            function = '__Pyx_SetItemInt_ByteArray'
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('SetItemInt', 'ObjectHandling.c'))
            function = '__Pyx_SetItemInt'
        index_code = self.index.result()
    else:
        index_code = self.index.py_result()
        if self.base.type is dict_type:
            function = 'PyDict_SetItem'
        else:
            function = 'PyObject_SetItem'
    code.putln(code.error_goto_if_neg('%s(%s, %s, %s%s)' % (function, self.base.py_result(), index_code, value_code, self.extra_index_params(code)), self.pos))