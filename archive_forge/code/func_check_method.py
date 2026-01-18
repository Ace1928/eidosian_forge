import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
def check_method(self, test_impl, args, expected, block_count, expects_inlined=True):
    j_func = njit(pipeline_class=IRPreservingTestPipeline)(test_impl)
    self.assertEqual(j_func(*args), expected)
    fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
    fir.blocks = fir.blocks
    self.assertEqual(len(fir.blocks), block_count)
    if expects_inlined:
        for block in fir.blocks.values():
            calls = list(block.find_exprs('call'))
            self.assertFalse(calls)
    else:
        allcalls = []
        for block in fir.blocks.values():
            allcalls += list(block.find_exprs('call'))
        self.assertTrue(allcalls)