from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
def get_buffer_by_name(self, buffer_name: str) -> Buffer | None:
    """
        Look in the layout for a buffer with the given name.
        Return `None` when nothing was found.
        """
    for w in self.walk():
        if isinstance(w, Window) and isinstance(w.content, BufferControl):
            if w.content.buffer.name == buffer_name:
                return w.content.buffer
    return None