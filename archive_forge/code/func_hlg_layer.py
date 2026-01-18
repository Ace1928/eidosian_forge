from __future__ import annotations
import contextlib
import importlib
import time
from typing import TYPE_CHECKING
def hlg_layer(hlg: HighLevelGraph, prefix: str) -> Layer:
    """Get the first layer from a HighLevelGraph whose name starts with a prefix"""
    for key, lyr in hlg.layers.items():
        if key.startswith(prefix):
            return lyr
    raise KeyError(f'No layer starts with {prefix!r}: {list(hlg.layers)}')