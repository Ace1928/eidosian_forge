import multiprocessing
import platform
import threading
import pickle
import weakref
from itertools import chain
from io import StringIO
import numpy as np
from numba import njit, jit, typeof, vectorize
from numba.core import types, errors
from numba import _dispatcher
from numba.tests.support import TestCase, captured_stdout
from numba.np.numpy_support import as_dtype
from numba.core.dispatcher import Dispatcher
from numba.extending import overload
from numba.tests.support import needs_lapack, SerialMixin
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
import unittest
@unittest.skipIf(_is_armv7l, 'Unaligned loads unsupported')
def test_immutability_in_array_dispatch(self):

    def foo(a):
        return np.sum(a)
    jitfoo = jit(nopython=True)(foo)
    n = 64
    r = int(np.sqrt(n))
    dt = np.int8
    count = np.complex128().itemsize // dt().itemsize
    tmp = np.arange(n * count + 1, dtype=dt)
    C_contig_aligned = tmp[:-1].view(np.complex128).reshape(r, r)
    C_contig_misaligned = tmp[1:].view(np.complex128).reshape(r, r)
    F_contig_aligned = C_contig_aligned.T
    F_contig_misaligned = C_contig_misaligned.T

    def check(name, a, disable_write_bit=False):
        a[:, :] = np.arange(n, dtype=np.complex128).reshape(r, r)
        if disable_write_bit:
            a.flags.writeable = False
        expected = foo(a)
        got = jitfoo(a)
        np.testing.assert_allclose(expected, got)
    check('C_contig_aligned', C_contig_aligned, disable_write_bit=True)
    check('F_contig_aligned', F_contig_aligned, disable_write_bit=True)
    check('C_contig_misaligned', C_contig_misaligned, disable_write_bit=True)
    check('F_contig_misaligned', F_contig_misaligned, disable_write_bit=True)