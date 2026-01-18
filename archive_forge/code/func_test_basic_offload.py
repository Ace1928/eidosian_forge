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
def test_basic_offload(self):
    _DEBUG = False

    @overload(np.sin, target='dpu')
    def ol_np_sin_DPU(x):

        def dpu_sin_impl(x):
            return 314159.0
        return dpu_sin_impl

    @djit(nopython=True)
    def foo(x):
        return np.sin(x)
    self.assertPreciseEqual(foo(5), 314159.0)

    @njit
    def foo(x):
        return np.sin(x)
    self.assertPreciseEqual(foo(5), np.sin(5))

    @register_pass(mutates_CFG=False, analysis_only=False)
    class DispatcherSwitcher(FunctionPass):
        _name = 'DispatcherSwitcher'

        def __init__(self):
            FunctionPass.__init__(self)

        def run_pass(self, state):
            func_ir = state.func_ir
            mutated = False
            for blk in func_ir.blocks.values():
                for call in blk.find_exprs('call'):
                    function = state.typemap[call.func.name]
                    tname = 'dpu'
                    with target_override(tname):
                        try:
                            sig = function.get_call_type(state.typingctx, state.calltypes[call].args, {})
                            disp = resolve_dispatcher_from_str(tname)
                            hw_ctx = disp.targetdescr.target_context
                            hw_ctx.get_function(function, sig)
                        except Exception as e:
                            if _DEBUG:
                                msg = f'Failed to find and compile an overload for {function} for {tname} due to {e}'
                                print(msg)
                            continue
                        hw_ctx._codelib_stack = state.targetctx._codelib_stack
                        call.target = tname
                        mutated = True
            return mutated

    class DPUOffloadCompiler(CompilerBase):

        def define_pipelines(self):
            pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
            pm.add_pass_after(DispatcherSwitcher, PreLowerStripPhis)
            pm.finalize()
            return [pm]

    @njit(pipeline_class=DPUOffloadCompiler)
    def foo(x):
        return (np.sin(x), np.cos(x))
    self.assertPreciseEqual(foo(5), (314159.0, np.cos(5)))