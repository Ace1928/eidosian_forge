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
def check_parfors_unsupported_prange_warning(self, warn_list):
    msg = 'prange or pndindex loop will not be executed in parallel due to there being more than one entry to or exit from the loop (e.g., an assertion).'
    warning_found = False
    for w in warn_list:
        if msg in str(w.message):
            warning_found = True
            break
    self.assertTrue(warning_found, 'Warning message should be found.')