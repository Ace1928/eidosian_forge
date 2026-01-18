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
@skip_unless_cffi
def test_cffi_function_pointer_template_source(self):
    from numba.tests import cffi_usecases as mod
    mod.init()
    func = mod.cffi_cos

    @njit
    def foo():
        func('a')
    with self.assertRaises(errors.TypingError) as raises:
        foo()
    excstr = str(raises.exception)
    self.assertIn("Type Restricted Function in function 'unknown'", excstr)