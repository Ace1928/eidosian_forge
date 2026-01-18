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
def generate_acquire_memoryviewslice(self, rhs, code):
    """
        Slices, coercions from objects, return values etc are new references.
        We have a borrowed reference in case of dst = src
        """
    from . import MemoryView
    MemoryView.put_acquire_memoryviewslice(lhs_cname=self.result(), lhs_type=self.type, lhs_pos=self.pos, rhs=rhs, code=code, have_gil=not self.in_nogil_context, first_assignment=self.cf_is_null)