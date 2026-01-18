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
def test_dict_iters(self):
    """Test that a List can be created from Dict iterators."""

    def generate_function(line):
        context = {}
        code = dedent('\n                from numba.typed import List, Dict\n                def bar():\n                    d = Dict()\n                    d[0], d[1], d[2] = "a", "b", "c"\n                    {}\n                    return l\n                ').format(line)
        exec(code, context)
        return njit(context['bar'])

    def generate_expected(values):
        expected = List()
        for i in values:
            expected.append(i)
        return expected
    for line, values in (('l = List(d)', (0, 1, 2)), ('l = List(d.keys())', (0, 1, 2)), ('l = List(d.values())', ('a', 'b', 'c')), ('l = List(d.items())', ((0, 'a'), (1, 'b'), (2, 'c')))):
        foo, expected = (generate_function(line), generate_expected(values))
        for func in (foo, foo.py_func):
            self.assertEqual(func(), expected)