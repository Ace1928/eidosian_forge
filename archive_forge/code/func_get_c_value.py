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
def get_c_value(self, builder, typ, name, dllimport=False):
    """
        Get a global value through its C-accessible *name*, with the given
        LLVM type.
        If *dllimport* is true, the symbol will be marked as imported
        from a DLL (necessary for AOT compilation under Windows).
        """
    module = builder.function.module
    try:
        gv = module.globals[name]
    except KeyError:
        gv = cgutils.add_global_variable(module, typ, name)
        if dllimport and self.aot_mode and (sys.platform == 'win32'):
            gv.storage_class = 'dllimport'
    return gv