import itertools
import numpy as np
import sys
from collections import namedtuple
from io import StringIO
from numba import njit, typeof, prange
from numba.core import (
from numba.tests.support import (TestCase, tag, skip_parfors_unsupported,
from numba.parfors.array_analysis import EquivSet, ArrayAnalysis
from numba.core.compiler import Compiler, Flags, PassManager
from numba.core.ir_utils import remove_dead
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
from numba.experimental import jitclass
import unittest
def compare_ir(self, ir_list):
    outputs = []
    for func_ir in ir_list:
        remove_dead(func_ir.blocks, func_ir.arg_names, func_ir)
        output = StringIO()
        func_ir.dump(file=output)
        outputs.append(output.getvalue())
    self.assertTrue(len(set(outputs)) == 1)