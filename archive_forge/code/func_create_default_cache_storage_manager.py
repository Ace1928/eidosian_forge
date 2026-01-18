from __future__ import annotations
from streamlit.runtime.caching.storage import CacheStorageManager
from streamlit.runtime.caching.storage.local_disk_cache_storage import (
def create_default_cache_storage_manager() -> CacheStorageManager:
    """
    Get the cache storage manager.
    It would be used both in server.py and in cli.py to have unified cache storage

    Returns
    -------
    CacheStorageManager
        The cache storage manager.

    """
    return LocalDiskCacheStorageManager()