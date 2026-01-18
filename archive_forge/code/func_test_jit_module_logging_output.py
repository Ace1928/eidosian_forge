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
def test_jit_module_logging_output(self):
    logger = logging.getLogger('numba.core.decorators')
    logger.setLevel(logging.DEBUG)
    jit_options = {'nopython': True, 'error_model': 'numpy'}
    with captured_logs(logger) as logs:
        with create_temp_module(self.source_lines, **jit_options) as test_module:
            logs = logs.getvalue()
            expected = ['Auto decorating function', 'from module {}'.format(test_module.__name__), 'with jit and options: {}'.format(jit_options)]
            self.assertTrue(all((i in logs for i in expected)))