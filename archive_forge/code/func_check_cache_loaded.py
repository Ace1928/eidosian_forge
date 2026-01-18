import sys
import os.path
import re
import subprocess
import numpy as np
from numba.tests.support import capture_cache_log
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest
def check_cache_loaded(self, cachelog, count):
    """
        Check number of cache-load were issued
        """
    data_loaded = self.regex_data_loaded.findall(cachelog)
    index_loaded = self.regex_index_loaded.findall(cachelog)
    self.assertEqual(len(data_loaded), count)
    self.assertEqual(len(index_loaded), count)