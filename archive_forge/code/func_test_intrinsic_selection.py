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
def test_intrinsic_selection(self):
    """
        Test to make sure that targets can share generic implementations and
        cannot reach implementations that are not in their target hierarchy.
        """

    @intrinsic(target='generic')
    def intrin_math_generic(tyctx, x, y):
        sig = x(x, y)

        def codegen(cgctx, builder, tyargs, llargs):
            return builder.mul(*llargs)
        return (sig, codegen)

    @intrinsic(target='dpu')
    def intrin_math_dpu(tyctx, x, y):
        sig = x(x, y)

        def codegen(cgctx, builder, tyargs, llargs):
            return builder.sub(*llargs)
        return (sig, codegen)

    @intrinsic(target='cpu')
    def intrin_math_cpu(tyctx, x, y):
        sig = x(x, y)

        def codegen(cgctx, builder, tyargs, llargs):
            return builder.add(*llargs)
        return (sig, codegen)

    @njit
    def cpu_foo_specific():
        return intrin_math_cpu(3, 4)
    self.assertEqual(cpu_foo_specific(), 7)

    @njit
    def cpu_foo_generic():
        return intrin_math_generic(3, 4)
    self.assertEqual(cpu_foo_generic(), 12)

    @njit
    def cpu_foo_dpu():
        return intrin_math_dpu(3, 4)
    accept = (errors.UnsupportedError, errors.TypingError)
    with self.assertRaises(accept) as raises:
        cpu_foo_dpu()
    msgs = ['Function resolution cannot find any matches for function', 'intrinsic intrin_math_dpu', 'for the current target']
    for msg in msgs:
        self.assertIn(msg, str(raises.exception))

    @djit(nopython=True)
    def dpu_foo_specific():
        return intrin_math_dpu(3, 4)
    self.assertEqual(dpu_foo_specific(), -1)

    @djit(nopython=True)
    def dpu_foo_generic():
        return intrin_math_generic(3, 4)
    self.assertEqual(dpu_foo_generic(), 12)

    @djit(nopython=True)
    def dpu_foo_cpu():
        return intrin_math_cpu(3, 4)
    accept = (errors.UnsupportedError, errors.TypingError)
    with self.assertRaises(accept) as raises:
        dpu_foo_cpu()
    msgs = ['Function resolution cannot find any matches for function', 'intrinsic intrin_math_cpu', 'for the current target']
    for msg in msgs:
        self.assertIn(msg, str(raises.exception))