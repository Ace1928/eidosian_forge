import time
from . import debug, errors, osutils, revision, trace
def note_missing_key(self, key):
    """Note that key is a missing key."""
    if self._cache_misses:
        self.missing_keys.add(key)