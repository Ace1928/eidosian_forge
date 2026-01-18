import ctypes
import itertools
import pickle
import random
import typing as pt
import unittest
from collections import OrderedDict
import numpy as np
from numba import (boolean, deferred_type, float32, float64, int16, int32,
from numba.core import errors, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import LoweringError, TypingError
from numba.core.runtime.nrt import MemInfo
from numba.experimental import jitclass
from numba.experimental.jitclass import _box
from numba.experimental.jitclass.base import JitClassType
from numba.tests.support import MemoryLeakMixin, TestCase, skip_if_typeguard
from numba.tests.support import skip_unless_scipy
def test_jitclass_longlabel_not_truncated(self):
    alphabet = [chr(ord('a') + x) for x in range(26)]
    spec = [(letter * 10, float64) for letter in alphabet]
    spec.extend([(letter.upper() * 10, float64) for letter in alphabet])

    @jitclass(spec)
    class TruncatedLabel(object):

        def __init__(self):
            self.aaaaaaaaaa = 10.0

        def meth1(self):
            self.bbbbbbbbbb = random.gauss(self.aaaaaaaaaa, self.aaaaaaaaaa)

        def meth2(self):
            self.meth1()
    TruncatedLabel().meth2()