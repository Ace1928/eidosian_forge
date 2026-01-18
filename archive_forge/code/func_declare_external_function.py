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
def declare_external_function(self, module, fndesc):
    fnty = self.get_external_function_type(fndesc)
    fn = cgutils.get_or_insert_function(module, fnty, fndesc.mangled_name)
    assert fn.is_declaration
    for ak, av in zip(fndesc.args, fn.args):
        av.name = 'arg.%s' % ak
    return fn