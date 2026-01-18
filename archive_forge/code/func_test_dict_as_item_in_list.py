import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
def test_dict_as_item_in_list(self):

    @njit
    def foo():
        l = List.empty_list(Dict.empty(int32, int32))
        d = Dict.empty(int32, int32)
        d[0] = 1
        l.append(d)
        return get_refcount(d)
    c = foo()
    if config.LLVM_REFPRUNE_PASS:
        self.assertEqual(1, c)
    else:
        self.assertEqual(2, c)