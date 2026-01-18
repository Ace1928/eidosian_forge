import sys
import subprocess
import numpy as np
import os
import warnings
from numba import jit, njit, types
from numba.core import errors
from numba.experimental import structref
from numba.extending import (overload, intrinsic, overload_method,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, DeadCodeElimination,
from numba.core.compiler_machinery import PassManager
from numba.core.types.functions import _err_reasons as error_reasons
from numba.tests.support import (skip_parfors_unsupported, override_config,
import unittest
def test_intrinsic_template_source(self):
    given_reason1 = 'x must be literal'
    given_reason2 = 'array.ndim must be 1'

    @intrinsic
    def myintrin(typingctx, x, arr):
        if not isinstance(x, types.IntegerLiteral):
            raise errors.RequireLiteralValue(given_reason1)
        if arr.ndim != 1:
            raise errors.NumbaValueError(given_reason2)
        sig = types.intp(x, arr)

        def codegen(context, builder, signature, args):
            pass
        return (sig, codegen)

    @njit
    def call_intrin():
        arr = np.zeros((2, 2))
        myintrin(1, arr)
    with self.assertRaises(errors.TypingError) as raises:
        call_intrin()
    excstr = str(raises.exception)
    self.assertIn(error_reasons['specific_error'].splitlines()[0], excstr)
    self.assertIn(given_reason1, excstr)
    self.assertIn(given_reason2, excstr)
    self.assertIn('Intrinsic in function', excstr)