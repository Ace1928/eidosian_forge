from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING, Callable
from prompt_toolkit.cache import FastDictCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.utils import get_cwidth
def draw_with_z_index(self, z_index: int, draw_func: Callable[[], None]) -> None:
    """
        Add a draw-function for a `Window` which has a >= 0 z_index.
        This will be postponed until `draw_all_floats` is called.
        """
    self._draw_float_functions.append((z_index, draw_func))