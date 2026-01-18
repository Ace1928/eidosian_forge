import os
import stat
import time
from .. import atomicfile, errors
from .. import filters as _mod_filters
from .. import osutils, trace
def cache_file_name(self):
    return self._cache_file_name