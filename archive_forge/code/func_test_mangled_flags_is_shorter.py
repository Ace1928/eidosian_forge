import re
from numba import njit
from numba.core.extending import overload
from numba.core.targetconfig import ConfigStack
from numba.core.compiler import Flags, DEFAULT_FLAGS
from numba.core import types
from numba.core.funcdesc import default_mangler
from numba.tests.support import TestCase, unittest
def test_mangled_flags_is_shorter(self):
    flags = Flags()
    flags.nrt = True
    flags.auto_parallel = True
    self.assertLess(len(flags.get_mangle_string()), len(flags.summary()))