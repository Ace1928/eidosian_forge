from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def create_state_temp_if_needed(self, pos, state, body):
    from .ParseTreeTransforms import YieldNodeCollector
    collector = YieldNodeCollector()
    collector.visitchildren(body)
    if not collector.yields:
        return
    if state == 'gil':
        temp_type = PyrexTypes.c_gilstate_type
    else:
        temp_type = PyrexTypes.c_threadstate_ptr_type
    from . import ExprNodes
    self.state_temp = ExprNodes.TempNode(pos, temp_type)