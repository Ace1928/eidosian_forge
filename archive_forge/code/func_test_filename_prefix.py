import sys
import os.path
import re
import subprocess
import numpy as np
from numba.tests.support import capture_cache_log
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest
def test_filename_prefix(self):
    mod = self.import_module()
    usecase = getattr(mod, 'direct_gufunc_cache_usecase')
    with capture_cache_log() as out:
        usecase()
    cachelog = out.getvalue()
    fmt1 = _fix_raw_path('/__pycache__/guf-{}')
    prefixed = re.findall(fmt1.format(self.modname), cachelog)
    fmt2 = _fix_raw_path('/__pycache__/{}')
    normal = re.findall(fmt2.format(self.modname), cachelog)
    self.assertGreater(len(normal), 2)
    self.assertEqual(len(normal), len(prefixed))