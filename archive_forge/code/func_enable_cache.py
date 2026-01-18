import time
from . import debug, errors, osutils, revision, trace
def enable_cache(self, cache_misses=True):
    """Enable cache."""
    if self._cache is not None:
        raise AssertionError('Cache enabled when already enabled.')
    self._cache = {}
    self._cache_misses = cache_misses
    self.missing_keys = set()