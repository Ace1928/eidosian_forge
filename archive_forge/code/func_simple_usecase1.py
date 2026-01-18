import collections
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
def simple_usecase1(rec):
    a = rec('a')
    b = rec('b')
    c = rec('c')
    a = b + c
    rec.mark('--1--')
    d = a + a
    rec.mark('--2--')
    return d