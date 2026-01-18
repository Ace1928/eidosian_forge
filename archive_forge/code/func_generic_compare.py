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
def generic_compare(self, builder, key, argtypes, args):
    """
        Compare the given LLVM values of the given Numba types using
        the comparison *key* (e.g. '==').  The values are first cast to
        a common safe conversion type.
        """
    at, bt = argtypes
    av, bv = args
    ty = self.typing_context.unify_types(at, bt)
    assert ty is not None
    cav = self.cast(builder, av, at, ty)
    cbv = self.cast(builder, bv, bt, ty)
    fnty = self.typing_context.resolve_value_type(key)
    cmpsig = fnty.get_call_type(self.typing_context, (ty, ty), {})
    cmpfunc = self.get_function(fnty, cmpsig)
    self.add_linking_libs(getattr(cmpfunc, 'libs', ()))
    return cmpfunc(builder, (cav, cbv))