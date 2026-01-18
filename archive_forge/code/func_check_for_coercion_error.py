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
def check_for_coercion_error(self, dst_type, env, fail=False, default=None):
    if fail and (not default):
        default = "Cannot assign type '%(FROM)s' to '%(TO)s'"
    message = find_coercion_error((self.type, dst_type), default, env)
    if message is not None:
        error(self.pos, message % {'FROM': self.type, 'TO': dst_type})
        return True
    if fail:
        self.fail_assignment(dst_type)
        return True
    return False