from __future__ import annotations
import inspect
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import Callable
def set_documentation_group(m):
    """A no-op for backwards compatibility of custom components published prior to 4.16.0"""
    pass