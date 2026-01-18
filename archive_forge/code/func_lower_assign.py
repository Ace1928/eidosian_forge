import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def lower_assign(self, inst):
    """
        The returned object must have a new reference
        """
    value = inst.value
    if isinstance(value, (ir.Const, ir.FreeVar)):
        return self.lower_const(value.value)
    elif isinstance(value, ir.Var):
        val = self.loadvar(value.name)
        self.incref(val)
        return val
    elif isinstance(value, ir.Expr):
        return self.lower_expr(value)
    elif isinstance(value, ir.Global):
        return self.lower_global(value.name, value.value)
    elif isinstance(value, ir.Yield):
        return self.lower_yield(value)
    elif isinstance(value, ir.Arg):
        param = self.func_ir.func_id.pysig.parameters.get(value.name)
        obj = self.fnargs[value.index]
        slot = cgutils.alloca_once_value(self.builder, obj)
        if param is not None and param.default is inspect.Parameter.empty:
            self.incref(obj)
            self.builder.store(obj, slot)
        else:
            typobj = self.pyapi.get_type(obj)
            is_omitted = self.builder.icmp_unsigned('==', typobj, self._omitted_typobj)
            with self.builder.if_else(is_omitted, likely=False) as (omitted, present):
                with present:
                    self.incref(obj)
                    self.builder.store(obj, slot)
                with omitted:
                    obj = self.pyapi.object_getattr_string(obj, 'value')
                    self.builder.store(obj, slot)
        return self.builder.load(slot)
    else:
        raise NotImplementedError(type(value), value)