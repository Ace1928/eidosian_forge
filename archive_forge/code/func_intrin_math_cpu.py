import unittest
from numba.tests.support import TestCase
import ctypes
import operator
from functools import cached_property
import numpy as np
from numba import njit, types
from numba.extending import overload, intrinsic, overload_classmethod
from numba.core.target_extension import (
from numba.core import utils, fastmathpass, errors
from numba.core.dispatcher import Dispatcher
from numba.core.descriptors import TargetDescriptor
from numba.core import cpu, typing, cgutils
from numba.core.base import BaseContext
from numba.core.compiler_lock import global_compiler_lock
from numba.core import callconv
from numba.core.codegen import CPUCodegen, JITCodeLibrary
from numba.core.callwrapper import PyCallWrapper
from numba.core.imputils import RegistryLoader, Registry
from numba import _dynfunc
import llvmlite.binding as ll
from llvmlite import ir as llir
from numba.core.runtime import rtsys
from numba.core import compiler
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.typed_passes import PreLowerStripPhis
@intrinsic(target='cpu')
def intrin_math_cpu(tyctx, x, y):
    sig = x(x, y)

    def codegen(cgctx, builder, tyargs, llargs):
        return builder.add(*llargs)
    return (sig, codegen)