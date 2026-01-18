from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
@property
def previous_control(self) -> UIControl:
    """
        Get the :class:`.UIControl` to previously had the focus.
        """
    try:
        return self._stack[-2].content
    except IndexError:
        return self._stack[-1].content