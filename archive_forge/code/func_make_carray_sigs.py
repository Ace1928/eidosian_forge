import ctypes
import os
import subprocess
import sys
from collections import namedtuple
import numpy as np
from numba import cfunc, carray, farray, njit
from numba.core import types, typing, utils
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import (TestCase, skip_unless_cffi, tag,
import unittest
from numba.np import numpy_support
def make_carray_sigs(self, formal_sig):
    """
        Generate a bunch of concrete signatures by varying the width
        and signedness of size arguments (see issue #1923).
        """
    for actual_size in (types.intp, types.int32, types.intc, types.uintp, types.uint32, types.uintc):
        args = tuple((actual_size if a == types.intp else a for a in formal_sig.args))
        yield formal_sig.return_type(*args)