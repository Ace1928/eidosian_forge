from __future__ import annotations
import os
import pickle
import time
from typing import TYPE_CHECKING
from fsspec.utils import atomic_write
def clear_expired(self, expiry_time: int) -> tuple[list[str], bool]:
    """Remove expired metadata from the cache.

        Returns names of files corresponding to expired metadata and a boolean
        flag indicating whether the writable cache is empty. Caller is
        responsible for deleting the expired files.
        """
    expired_files = []
    for path, detail in self.cached_files[-1].copy().items():
        if time.time() - detail['time'] > expiry_time:
            fn = detail.get('fn', '')
            if not fn:
                raise RuntimeError(f"Cache metadata does not contain 'fn' for {path}")
            fn = os.path.join(self._storage[-1], fn)
            expired_files.append(fn)
            self.cached_files[-1].pop(path)
    if self.cached_files[-1]:
        cache_path = os.path.join(self._storage[-1], 'cache')
        self._save(self.cached_files[-1], cache_path)
    writable_cache_empty = not self.cached_files[-1]
    return (expired_files, writable_cache_empty)