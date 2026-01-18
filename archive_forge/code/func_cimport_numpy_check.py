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
def cimport_numpy_check(node, code):
    for mod in code.globalstate.module_node.scope.cimported_modules:
        if mod.name != node.module_name:
            continue
        import_array = mod.lookup_here('import_array')
        _import_array = mod.lookup_here('_import_array')
        used = import_array and import_array.used or (_import_array and _import_array.used)
        if (import_array or _import_array) and (not used):
            if _import_array and _import_array.type.is_cfunction:
                warning(node.pos, "'numpy.import_array()' has been added automatically since 'numpy' was cimported but 'numpy.import_array' was not called.", 0)
                code.globalstate.use_utility_code(UtilityCode.load_cached('NumpyImportArray', 'NumpyImportArray.c'))
                return