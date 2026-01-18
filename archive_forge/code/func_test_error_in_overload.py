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
def test_error_in_overload(self):

    def bar(x):
        pass

    @overload(bar)
    def ol_bar(x):
        x.some_invalid_attr

        def impl(x):
            pass
        return impl
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', errors.NumbaPendingDeprecationWarning)
        for style, err_class in (('new_style', AttributeError), ('old_style', errors.TypingError)):
            with override_config('CAPTURED_ERRORS', style):
                with self.assertRaises(err_class) as raises:

                    @njit('void(int64)')
                    def foo(x):
                        bar(x)
                expected = "object has no attribute 'some_invalid_attr'"
                self.assertIn(expected, str(raises.exception))