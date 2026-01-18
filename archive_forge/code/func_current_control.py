from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
@current_control.setter
def current_control(self, control: UIControl) -> None:
    """
        Set the :class:`.UIControl` to receive the focus.
        """
    for window in self.find_all_windows():
        if window.content == control:
            self.current_window = window
            return
    raise ValueError('Control not found in the user interface.')