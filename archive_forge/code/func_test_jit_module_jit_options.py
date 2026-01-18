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
def test_jit_module_jit_options(self):
    jit_options = {'nopython': True, 'nogil': False, 'error_model': 'numpy', 'boundscheck': False}
    with create_temp_module(self.source_lines, **jit_options) as test_module:
        self.assertEqual(test_module.inc.targetoptions, jit_options)