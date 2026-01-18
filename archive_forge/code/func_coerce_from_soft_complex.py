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
def coerce_from_soft_complex(arg, dst_type, env):
    from .UtilNodes import HasGilNode
    cfunc_type = PyrexTypes.CFuncType(PyrexTypes.c_double_type, [PyrexTypes.CFuncTypeArg('value', PyrexTypes.soft_complex_type, None), PyrexTypes.CFuncTypeArg('have_gil', PyrexTypes.c_bint_type, None)], exception_value='-1', exception_check=True, nogil=True)
    call = PythonCapiCallNode(arg.pos, '__Pyx_SoftComplexToDouble', cfunc_type, utility_code=UtilityCode.load_cached('SoftComplexToDouble', 'Complex.c'), args=[arg, HasGilNode(arg.pos)])
    call = call.analyse_types(env)
    if call.type != dst_type:
        call = call.coerce_to(dst_type, env)
    return call