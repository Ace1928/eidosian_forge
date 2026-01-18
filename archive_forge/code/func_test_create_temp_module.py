from collections import namedtuple
import gc
import os
import operator
import sys
import weakref
import numpy as np
from numba.core import types, typing, errors, sigutils
from numba.core.types.abstract import _typecache
from numba.core.types.functions import _header_lead
from numba.core.typing.templates import make_overload_template
from numba import jit, njit, typeof
from numba.core.extending import (overload, register_model, models, unbox,
from numba.tests.support import TestCase, create_temp_module
from numba.tests.enum_usecases import Color, Shake, Shape
import unittest
from numba.np import numpy_support
from numba.core import types
def test_create_temp_module(self):
    sys_path_original = list(sys.path)
    sys_modules_original = dict(sys.modules)
    with create_temp_module(self.source_lines) as test_module:
        temp_module_dir = os.path.dirname(test_module.__file__)
        self.assertEqual(temp_module_dir, sys.path[0])
        self.assertEqual(sys.path[1:], sys_path_original)
        self.assertTrue(test_module.__name__ in sys.modules)
    self.assertEqual(sys.path, sys_path_original)
    self.assertEqual(sys.modules, sys_modules_original)