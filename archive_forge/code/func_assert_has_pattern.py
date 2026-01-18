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
def assert_has_pattern(self, fullname, text):
    pat = self.make_pattern(fullname)
    self.assertIsNotNone(re.search(pat, text), msg='expected {}'.format(pat))