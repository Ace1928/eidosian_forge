import os
import platform
import re
import textwrap
import warnings
import numpy as np
from numba.tests.support import (TestCase, override_config, override_env_config,
from numba import jit, njit
from numba.core import types, compiler, utils
from numba.core.errors import NumbaPerformanceWarning
from numba import prange
from numba.experimental import jitclass
import unittest
def check_parfors_warning(self, warn_list):
    msg = "'parallel=True' was specified but no transformation for parallel execution was possible."
    warning_found = False
    for w in warn_list:
        if msg in str(w.message):
            warning_found = True
            break
    self.assertTrue(warning_found, 'Warning message should be found.')