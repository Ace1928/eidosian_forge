import sys
import os.path
import re
import subprocess
import numpy as np
from numba.tests.support import capture_cache_log
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest
def check_cache_saved(self, cachelog, count):
    """
        Check number of cache-save were issued
        """
    data_saved = self.regex_data_saved.findall(cachelog)
    index_saved = self.regex_index_saved.findall(cachelog)
    self.assertEqual(len(data_saved), count)
    self.assertEqual(len(index_saved), count)