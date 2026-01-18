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
def generate_yield_code(self, code):
    """
        Generate the code to return the argument in 'Naming.retval_cname'
        and to continue at the yield label.
        """
    label_num, label_name = code.new_yield_label(self.expr_keyword.replace(' ', '_'))
    code.use_label(label_name)
    saved = []
    code.funcstate.closure_temps.reset()
    for cname, type, manage_ref in code.funcstate.temps_in_use():
        save_cname = code.funcstate.closure_temps.allocate_temp(type)
        saved.append((cname, save_cname, type))
        if type.is_cpp_class:
            code.globalstate.use_utility_code(UtilityCode.load_cached('MoveIfSupported', 'CppSupport.cpp'))
            cname = '__PYX_STD_MOVE_IF_SUPPORTED(%s)' % cname
        else:
            code.put_xgiveref(cname, type)
        code.putln('%s->%s = %s;' % (Naming.cur_scope_cname, save_cname, cname))
    code.put_xgiveref(Naming.retval_cname, py_object_type)
    profile = code.globalstate.directives['profile']
    linetrace = code.globalstate.directives['linetrace']
    if profile or linetrace:
        code.put_trace_return(Naming.retval_cname, nogil=not code.funcstate.gil_owned)
    code.put_finish_refcount_context()
    if code.funcstate.current_except is not None:
        code.putln('__Pyx_Coroutine_SwapException(%s);' % Naming.generator_cname)
    else:
        code.putln('__Pyx_Coroutine_ResetAndClearException(%s);' % Naming.generator_cname)
    code.putln('/* return from %sgenerator, %sing value */' % ('async ' if self.in_async_gen else '', 'await' if self.is_await else 'yield'))
    code.putln('%s->resume_label = %d;' % (Naming.generator_cname, label_num))
    if self.in_async_gen and (not self.is_await):
        code.putln('return __Pyx__PyAsyncGenValueWrapperNew(%s);' % Naming.retval_cname)
    else:
        code.putln('return %s;' % Naming.retval_cname)
    code.put_label(label_name)
    for cname, save_cname, type in saved:
        save_cname = '%s->%s' % (Naming.cur_scope_cname, save_cname)
        if type.is_cpp_class:
            save_cname = '__PYX_STD_MOVE_IF_SUPPORTED(%s)' % save_cname
        code.putln('%s = %s;' % (cname, save_cname))
        if type.is_pyobject:
            code.putln('%s = 0;' % save_cname)
            code.put_xgotref(cname, type)
        elif type.is_memoryviewslice:
            code.putln('%s.memview = NULL; %s.data = NULL;' % (save_cname, save_cname))
    self.generate_sent_value_handling_code(code, Naming.sent_value_cname)
    if self.result_is_used:
        self.allocate_temp_result(code)
        code.put('%s = %s; ' % (self.result(), Naming.sent_value_cname))
        code.put_incref(self.result(), py_object_type)