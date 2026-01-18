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
def generate_function_body(self, env, code):
    body_cname = self.gbody.entry.func_cname
    name = code.intern_identifier(self.name)
    qualname = code.intern_identifier(self.qualname)
    module_name = code.intern_identifier(self.module_name)
    code.putln('{')
    code.putln('__pyx_CoroutineObject *gen = __Pyx_%s_New((__pyx_coroutine_body_t) %s, %s, (PyObject *) %s, %s, %s, %s); %s' % (self.gen_type_name, body_cname, self.code_object.calculate_result_code(code) if self.code_object else 'NULL', Naming.cur_scope_cname, name, qualname, module_name, code.error_goto_if_null('gen', self.pos)))
    code.put_decref(Naming.cur_scope_cname, py_object_type)
    if self.requires_classobj:
        classobj_cname = 'gen->classobj'
        code.putln('%s = __Pyx_CyFunction_GetClassObj(%s);' % (classobj_cname, Naming.self_cname))
        code.put_incref(classobj_cname, py_object_type)
        code.put_giveref(classobj_cname, py_object_type)
    code.put_finish_refcount_context()
    code.putln('return (PyObject *) gen;')
    code.putln('}')