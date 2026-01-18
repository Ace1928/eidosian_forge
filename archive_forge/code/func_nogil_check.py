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
def nogil_check(self, env):
    names = ('start', 'stop', 'step', 'target')
    nodes = (self.start, self.stop, self.step, self.target)
    for name, node in zip(names, nodes):
        if node is not None and node.type.is_pyobject:
            error(node.pos, "%s may not be a Python object as we don't have the GIL" % name)