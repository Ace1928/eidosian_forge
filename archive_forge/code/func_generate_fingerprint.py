import inspect
import os
import random
import shutil
import tempfile
import weakref
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import xxhash
from . import config
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH
from .utils._dill import dumps
from .utils.deprecation_utils import deprecated
from .utils.logging import get_logger
def generate_fingerprint(dataset: 'Dataset') -> str:
    state = dataset.__dict__
    hasher = Hasher()
    for key in sorted(state):
        if key == '_fingerprint':
            continue
        hasher.update(key)
        hasher.update(state[key])
    for cache_file in dataset.cache_files:
        hasher.update(os.path.getmtime(cache_file['filename']))
    return hasher.hexdigest()