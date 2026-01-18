import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
@box(DummyType)
def box_dummy(typ, obj, c):
    clazobj = c.pyapi.unserialize(c.pyapi.serialize_object(Dummy))
    return c.pyapi.call_function_objargs(clazobj, ())