from __future__ import annotations
import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, Dict, Iterable, List, Optional, overload, Tuple, Union
import torch
from .common import amp_definitely_not_available
def set_backoff_factor(self, new_factor: float) -> None:
    """Set a new scale backoff factor.

        Args:
            new_scale (float):  Value to use as the new scale backoff factor.
        """
    self._backoff_factor = new_factor