from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
def get_focusable_windows(self) -> Iterable[Window]:
    """
        Return all the :class:`.Window` objects which are focusable (in the
        'modal' area).
        """
    for w in self.walk_through_modal_area():
        if isinstance(w, Window) and w.content.is_focusable():
            yield w