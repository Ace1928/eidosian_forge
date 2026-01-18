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
def analyse_result_type(self, env):
    true_val_type = self.true_val.type
    false_val_type = self.false_val.type
    self.type = PyrexTypes.independent_spanning_type(true_val_type, false_val_type)
    if self.type.is_reference:
        self.type = PyrexTypes.CFakeReferenceType(self.type.ref_base_type)
    if self.type.is_pyobject:
        self.result_ctype = py_object_type
    elif self.true_val.is_ephemeral() or self.false_val.is_ephemeral():
        error(self.pos, 'Unsafe C derivative of temporary Python reference used in conditional expression')
    if true_val_type.is_pyobject or false_val_type.is_pyobject or self.type.is_pyobject:
        if true_val_type != self.type:
            self.true_val = self.true_val.coerce_to(self.type, env)
        if false_val_type != self.type:
            self.false_val = self.false_val.coerce_to(self.type, env)
    if self.type.is_error:
        self.type_error()
    return self