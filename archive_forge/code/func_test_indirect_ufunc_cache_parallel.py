import sys
import os.path
import re
import subprocess
import numpy as np
from numba.tests.support import capture_cache_log
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest
def test_indirect_ufunc_cache_parallel(self):
    self.test_indirect_ufunc_cache(target='parallel')