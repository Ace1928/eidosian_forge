from __future__ import annotations
import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, Dict, Iterable, List, Optional, overload, Tuple, Union
import torch
from .common import amp_definitely_not_available
def set_growth_interval(self, new_interval: int) -> None:
    """Set a new growth interval.

        Args:
            new_interval (int):  Value to use as the new growth interval.
        """
    self._growth_interval = new_interval