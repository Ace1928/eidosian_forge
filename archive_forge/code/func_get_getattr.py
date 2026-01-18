from collections import defaultdict
import copy
import sys
from itertools import permutations, takewhile
from contextlib import contextmanager
from functools import cached_property
from llvmlite import ir as llvmir
from llvmlite.ir import Constant
import llvmlite.binding as ll
from numba.core import types, utils, datamodel, debuginfo, funcdesc, config, cgutils, imputils
from numba.core import event, errors, targetconfig
from numba import _dynfunc, _helperlib
from numba.core.compiler_lock import global_compiler_lock
from numba.core.pythonapi import PythonAPI
from numba.core.imputils import (user_function, user_generator,
from numba.cpython import builtins
def get_getattr(self, typ, attr):
    """
        Get the getattr() implementation for the given type and attribute name.
        The return value is a callable with the signature
        (context, builder, typ, val, attr).
        """
    const_attr = (typ, attr) not in self.nonconst_module_attrs
    is_module = isinstance(typ, types.Module)
    if is_module and const_attr:
        attrty = self.typing_context.resolve_module_constants(typ, attr)
        if attrty is None or isinstance(attrty, types.Dummy):
            return None
        else:
            pyval = getattr(typ.pymod, attr)

            def imp(context, builder, typ, val, attr):
                llval = self.get_constant_generic(builder, attrty, pyval)
                return impl_ret_borrowed(context, builder, attrty, llval)
            return imp
    overloads = self._getattrs[attr]
    try:
        return overloads.find((typ,))
    except errors.NumbaNotImplementedError:
        pass
    overloads = self._getattrs[None]
    try:
        return overloads.find((typ,))
    except errors.NumbaNotImplementedError:
        pass
    raise NotImplementedError('No definition for lowering %s.%s' % (typ, attr))