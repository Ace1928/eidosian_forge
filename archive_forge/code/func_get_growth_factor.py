from __future__ import annotations
import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, Dict, Iterable, List, Optional, overload, Tuple, Union
import torch
from .common import amp_definitely_not_available
def get_growth_factor(self) -> float:
    """Return a Python float containing the scale growth factor."""
    return self._growth_factor