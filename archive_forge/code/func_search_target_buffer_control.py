from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
@property
def search_target_buffer_control(self) -> BufferControl | None:
    """
        Return the :class:`.BufferControl` in which we are searching or `None`.
        """
    control = self.current_control
    if isinstance(control, SearchBufferControl):
        return self.search_links.get(control)
    else:
        return None