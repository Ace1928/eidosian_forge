import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
def codegen_loops_impl(self, loop_nest, code, worksharing):
    threads = parallel_num_threads()
    assert self.call_ranges is not None
    par_depth = self.decide_parallel_depth(self.call_ranges[:loop_nest.max_parallel_depth()], threads)
    with contextlib.ExitStack() as stack:
        if par_depth:
            if loop_nest.is_reduction_only():
                worksharing.close()
            else:
                worksharing.parallel(threads)
            loop_nest.mark_parallel(par_depth)
        elif threads > 1:
            if worksharing.single():
                stack.enter_context(code.indent())

        def gen_kernel(kernel):
            with contextlib.ExitStack() as stack:
                assert kernel
                if hasattr(kernel, 'codegen_inner_loops'):
                    code.splice(kernel.preloads)
                    kernel.codegen_inner_loops(code)
                    stack.enter_context(code.indent())
                code.splice(kernel.loads)
                code.splice(kernel.compute)
                code.splice(kernel.stores)
            if hasattr(kernel, 'codegen_inner_loops'):
                code.splice(kernel.poststores)

        def get_reduction_code_buffer(loops, is_suffix=True):
            for loop in loops:
                for kernel in loop.get_kernels():
                    if is_suffix:
                        return kernel.reduction_suffix
                    else:
                        return kernel.reduction_prefix
            return None

        def gen_loops(loops: List[LoopLevel], in_reduction=False):
            with contextlib.ExitStack() as stack_outer:
                if loops:
                    loop = loops[0]
                    if loop.is_reduction() and (not in_reduction):
                        reduction_prefix = get_reduction_code_buffer(loops, is_suffix=False)
                        if reduction_prefix:
                            stack_outer.enter_context(code.indent())
                        code.splice(reduction_prefix)
                    if loop_nest.is_reduction_only() and loop.parallel:
                        worksharing.parallel(threads)
                for loop in loops:
                    gen_loop(loop, in_reduction)
                if loops:
                    loop = loops[0]
                    if loop_nest.is_reduction_only() and loop.parallel:
                        worksharing.close()
                    if loop.is_reduction() and (not in_reduction):
                        code.splice(get_reduction_code_buffer(loops, is_suffix=True))

        def gen_loop(loop: LoopLevel, in_reduction=False):
            with contextlib.ExitStack() as stack:
                loop_lines = loop.lines()
                if loop_lines is None:
                    return
                code.writelines(loop_lines)
                stack.enter_context(code.indent())
                if loop.inner:
                    gen_loops(loop.inner, loop.is_reduction())
                else:
                    kernels = loop.get_kernels()
                    assert len(kernels) == 1
                    gen_kernel(kernels[0])
        stack.enter_context(code.indent())
        if loop_nest.root:
            gen_loops(loop_nest.root)
        else:
            gen_kernel(loop_nest.kernel)