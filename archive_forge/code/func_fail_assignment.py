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
def fail_assignment(self, dst_type):
    src_name = self.entry.name if hasattr(self, 'entry') else None
    src_resolved = " (alias of '{0}')".format(self.type.resolve()) if self.type.is_typedef else ''
    dst_resolved = " (alias of '{0}')".format(dst_type.resolve()) if dst_type.is_typedef else ''
    extra_diagnostics = dst_type.assignment_failure_extra_info(self.type, src_name)
    if extra_diagnostics:
        extra_diagnostics = '. ' + extra_diagnostics
    error(self.pos, "Cannot assign type '%s'%s to '%s'%s%s" % (self.type, src_resolved, dst_type, dst_resolved, extra_diagnostics))