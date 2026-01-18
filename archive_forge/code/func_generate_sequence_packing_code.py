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
def generate_sequence_packing_code(self, code, target=None, plain=False):
    if target is None:
        target = self.result()
    size_factor = c_mult = ''
    mult_factor = None
    if self.mult_factor and (not plain):
        mult_factor = self.mult_factor
        if mult_factor.type.is_int:
            c_mult = mult_factor.result()
            if isinstance(mult_factor.constant_result, _py_int_types) and mult_factor.constant_result > 0:
                size_factor = ' * %s' % mult_factor.constant_result
            elif mult_factor.type.signed:
                size_factor = ' * ((%s<0) ? 0:%s)' % (c_mult, c_mult)
            else:
                size_factor = ' * (%s)' % (c_mult,)
    if self.type is tuple_type and (self.is_literal or self.slow) and (not c_mult):
        code.putln('%s = PyTuple_Pack(%d, %s); %s' % (target, len(self.args), ', '.join((arg.py_result() for arg in self.args)), code.error_goto_if_null(target, self.pos)))
        code.put_gotref(target, py_object_type)
    elif self.type.is_ctuple:
        for i, arg in enumerate(self.args):
            code.putln('%s.f%s = %s;' % (target, i, arg.result()))
    else:
        if self.type is list_type:
            create_func, set_item_func = ('PyList_New', '__Pyx_PyList_SET_ITEM')
        elif self.type is tuple_type:
            create_func, set_item_func = ('PyTuple_New', '__Pyx_PyTuple_SET_ITEM')
        else:
            raise InternalError('sequence packing for unexpected type %s' % self.type)
        arg_count = len(self.args)
        code.putln('%s = %s(%s%s); %s' % (target, create_func, arg_count, size_factor, code.error_goto_if_null(target, self.pos)))
        code.put_gotref(target, py_object_type)
        if c_mult:
            counter = Naming.quick_temp_cname
            code.putln('{ Py_ssize_t %s;' % counter)
            if arg_count == 1:
                offset = counter
            else:
                offset = '%s * %s' % (counter, arg_count)
            code.putln('for (%s=0; %s < %s; %s++) {' % (counter, counter, c_mult, counter))
        else:
            offset = ''
        for i in range(arg_count):
            arg = self.args[i]
            if c_mult or not arg.result_in_temp():
                code.put_incref(arg.result(), arg.ctype())
            arg.generate_giveref(code)
            code.putln('if (%s(%s, %s, %s)) %s;' % (set_item_func, target, (offset and i) and '%s + %s' % (offset, i) or (offset or i), arg.py_result(), code.error_goto(self.pos)))
        if c_mult:
            code.putln('}')
            code.putln('}')
    if mult_factor is not None and mult_factor.type.is_pyobject:
        code.putln('{ PyObject* %s = PyNumber_InPlaceMultiply(%s, %s); %s' % (Naming.quick_temp_cname, target, mult_factor.py_result(), code.error_goto_if_null(Naming.quick_temp_cname, self.pos)))
        code.put_gotref(Naming.quick_temp_cname, py_object_type)
        code.put_decref(target, py_object_type)
        code.putln('%s = %s;' % (target, Naming.quick_temp_cname))
        code.putln('}')