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
def call_internal_no_propagate(self, builder, fndesc, sig, args):
    """Similar to `.call_internal()` but does not handle or propagate
        the return status automatically.
        """
    llvm_mod = builder.module
    fn = self.declare_function(llvm_mod, fndesc)
    status, res = self.call_conv.call_function(builder, fn, sig.return_type, sig.args, args)
    return (status, res)