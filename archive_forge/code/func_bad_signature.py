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
def bad_signature(self):
    sig = self.entry.signature
    expected_str = '%d' % sig.min_num_fixed_args()
    if sig.has_generic_args:
        expected_str += ' or more'
    elif sig.optional_object_arg_count:
        expected_str += ' to %d' % sig.max_num_fixed_args()
    name = self.name
    if name.startswith('__') and name.endswith('__'):
        desc = 'Special method'
    else:
        desc = 'Method'
    error(self.pos, '%s %s has wrong number of arguments (%d declared, %s expected)' % (desc, self.name, len(self.args), expected_str))