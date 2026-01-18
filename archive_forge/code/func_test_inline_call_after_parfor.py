import re
import numpy as np
from numba.tests.support import (TestCase, override_config, captured_stdout,
from numba import jit, njit
from numba.core import types, ir, postproc, compiler
from numba.core.ir_utils import (guard, find_callname, find_const,
from numba.core.registry import CPUDispatcher
from numba.core.inline_closurecall import inline_closure_call
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
import unittest
@skip_parfors_unsupported
def test_inline_call_after_parfor(self):
    from numba.tests.inlining_usecases import __dummy__

    def test_impl(A):
        __dummy__()
        return A.sum()
    j_func = njit(parallel=True, pipeline_class=InlineTestPipeline)(test_impl)
    A = np.arange(10)
    self.assertEqual(test_impl(A), j_func(A))