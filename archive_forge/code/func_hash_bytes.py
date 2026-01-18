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
@classmethod
def hash_bytes(cls, value: Union[bytes, List[bytes]]) -> str:
    value = [value] if isinstance(value, bytes) else value
    m = xxhash.xxh64()
    for x in value:
        m.update(x)
    return m.hexdigest()