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
def declare_optional_arg_struct(self, func_type, env, fused_cname=None):
    """
        Declares the optional argument struct (the struct used to hold the
        values for optional arguments). For fused cdef functions, this is
        deferred as analyse_declarations is called only once (on the fused
        cdef function).
        """
    scope = StructOrUnionScope()
    arg_count_member = '%sn' % Naming.pyrex_prefix
    scope.declare_var(arg_count_member, PyrexTypes.c_int_type, self.pos)
    for arg in func_type.args[len(func_type.args) - self.optional_arg_count:]:
        scope.declare_var(arg.name, arg.type, arg.pos, allow_pyobject=True, allow_memoryview=True)
    struct_cname = env.mangle(Naming.opt_arg_prefix, self.base.name)
    if fused_cname is not None:
        struct_cname = PyrexTypes.get_fused_cname(fused_cname, struct_cname)
    op_args_struct = env.global_scope().declare_struct_or_union(name=struct_cname, kind='struct', scope=scope, typedef_flag=0, pos=self.pos, cname=struct_cname)
    op_args_struct.defined_in_pxd = 1
    op_args_struct.used = 1
    func_type.op_arg_struct = PyrexTypes.c_ptr_type(op_args_struct.type)