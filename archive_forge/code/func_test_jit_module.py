import os
import sys
import inspect
import contextlib
import numpy as np
import logging
from io import StringIO
import unittest
from numba.tests.support import SerialMixin, create_temp_module
from numba.core import dispatcher
from numba import jit_module
import numpy as np
from numba import jit, jit_module
def test_jit_module(self):
    with create_temp_module(self.source_lines) as test_module:
        self.assertIsInstance(test_module.inc, dispatcher.Dispatcher)
        self.assertIsInstance(test_module.add, dispatcher.Dispatcher)
        self.assertIsInstance(test_module.inc_add, dispatcher.Dispatcher)
        self.assertTrue(test_module.mean is np.mean)
        self.assertTrue(inspect.isclass(test_module.Foo))
        x, y = (1.7, 2.3)
        self.assertEqual(test_module.inc(x), test_module.inc.py_func(x))
        self.assertEqual(test_module.add(x, y), test_module.add.py_func(x, y))
        self.assertEqual(test_module.inc_add(x), test_module.inc_add.py_func(x))