from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
@current_window.setter
def current_window(self, value: Window) -> None:
    """Set the :class:`.Window` object to be currently focused."""
    self._stack.append(value)