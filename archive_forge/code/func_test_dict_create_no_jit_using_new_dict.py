import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin, unittest,
from numba.experimental import jitclass
from numba.extending import overload
def test_dict_create_no_jit_using_new_dict(self):
    with override_config('DISABLE_JIT', True):
        with forbid_codegen():
            d = dictobject.new_dict(int32, float32)
            self.assertEqual(type(d), dict)