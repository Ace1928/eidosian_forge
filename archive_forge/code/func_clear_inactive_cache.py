import time
import os
import sys
import hashlib
import gc
import shutil
import platform
import logging
import warnings
import pickle
from pathlib import Path
from typing import Dict, Any
def clear_inactive_cache(cache_path=None, inactivity_threshold=_CACHED_FILE_MAXIMUM_SURVIVAL):
    if cache_path is None:
        cache_path = _default_cache_path
    if not cache_path.exists():
        return False
    for dirname in os.listdir(cache_path):
        version_path = cache_path.joinpath(dirname)
        if not version_path.is_dir():
            continue
        for file in os.scandir(version_path):
            if file.stat().st_atime + _CACHED_FILE_MAXIMUM_SURVIVAL <= time.time():
                try:
                    os.remove(file.path)
                except OSError:
                    continue
    else:
        return True