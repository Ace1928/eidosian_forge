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
def as_sliced_node(self, start, stop, step=None):
    value = type(self.value)(self.value[start:stop:step])
    value.encoding = self.value.encoding
    if self.unicode_value is not None:
        if StringEncoding.string_contains_surrogates(self.unicode_value[:stop]):
            return None
        unicode_value = StringEncoding.EncodedString(self.unicode_value[start:stop:step])
    else:
        unicode_value = None
    return StringNode(self.pos, value=value, unicode_value=unicode_value, constant_result=value, is_identifier=self.is_identifier)