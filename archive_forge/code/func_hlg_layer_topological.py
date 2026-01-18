from __future__ import annotations
import contextlib
import importlib
import time
from typing import TYPE_CHECKING
def hlg_layer_topological(hlg: HighLevelGraph, i: int) -> Layer:
    """Get the layer from a HighLevelGraph at position ``i``, topologically"""
    return hlg.layers[hlg._toposort_layers()[i]]