import sys
import os.path
import re
import subprocess
import numpy as np
from numba.tests.support import capture_cache_log
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest
def check_ufunc_cache(self, usecase_name, n_overloads, **kwargs):
    """
        Check number of cache load/save.
        There should be one per overloaded version.
        """
    mod = self.import_module()
    usecase = getattr(mod, usecase_name)
    with capture_cache_log() as out:
        new_ufunc = usecase(**kwargs)
    cachelog = out.getvalue()
    self.check_cache_saved(cachelog, count=n_overloads)
    with capture_cache_log() as out:
        cached_ufunc = usecase(**kwargs)
    cachelog = out.getvalue()
    self.check_cache_loaded(cachelog, count=n_overloads)
    return (new_ufunc, cached_ufunc)