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
def test_jit_module_logging_level(self):
    logger = logging.getLogger('numba.core.decorators')
    logger.setLevel(logging.INFO)
    with captured_logs(logger) as logs:
        with create_temp_module(self.source_lines):
            self.assertEqual(logs.getvalue(), '')