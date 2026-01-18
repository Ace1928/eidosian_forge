import importlib
from functools import wraps
from typing import Protocol, runtime_checkable
import numpy as np
from scipy.sparse import issparse
from .._config import get_config
from ._available_if import available_if
def check_library_installed(library):
    """Check library is installed."""
    try:
        return importlib.import_module(library)
    except ImportError as exc:
        raise ImportError(f"Setting output container to '{library}' requires {library} to be installed") from exc