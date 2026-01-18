from __future__ import annotations
import warnings
import numpy as np
import pyarrow
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
def fallback_performancewarning(version: str | None=None) -> None:
    """
    Raise a PerformanceWarning for falling back to ExtensionArray's
    non-pyarrow method
    """
    msg = 'Falling back on a non-pyarrow code path which may decrease performance.'
    if version is not None:
        msg += f' Upgrade to pyarrow >={version} to possibly suppress this warning.'
    warnings.warn(msg, PerformanceWarning, stacklevel=find_stack_level())