import importlib.util
import logging
import sys
import types
from typing import Any, Dict, List, Optional, Sequence, cast
import numpy as np
from ._typing import _T
def lazy_isinstance(instance: Any, module: str, name: str) -> bool:
    """Use string representation to identify a type."""
    cls = instance.__class__
    is_same_module = cls.__module__ == module
    has_same_name = cls.__name__ == name
    return is_same_module and has_same_name