import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
@linux_only
@x86_only
def test_inline_asm(self):
    """The InlineAsm class from llvmlite.ir has no 'name' attr the refcount
        pruning pass should be tolerant to this"""
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()

    @intrinsic
    def bar(tyctx, x, y):

        def codegen(cgctx, builder, sig, args):
            arg_0, arg_1 = args
            fty = ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(32)])
            mul = builder.asm(fty, 'mov $2, $0; imul $1, $0', '=&r,r,r', (arg_0, arg_1), name='asm_mul', side_effect=False)
            return impl_ret_untracked(cgctx, builder, sig.return_type, mul)
        return (signature(types.int32, types.int32, types.int32), codegen)

    @njit(['int32(int32)'])
    def foo(x):
        x += 1
        z = bar(x, 2)
        return z
    self.assertEqual(foo(10), 22)