import time
from . import debug, errors, osutils, revision, trace
def get_cached_map(self):
    """Return any cached get_parent_map values."""
    if self._cache is None:
        return None
    return dict(self._cache)