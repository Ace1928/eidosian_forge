from __future__ import annotations
import io
import os
import sys
from typing import Callable, Dict, Hashable, Iterable, Sequence, TextIO, Tuple
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.data_structures import Size
from prompt_toolkit.output import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES, Attrs
from prompt_toolkit.utils import is_dumb_terminal
from .color_depth import ColorDepth
from .flush_stdout import flush_stdout
def reset_cursor_key_mode(self) -> None:
    """
        For vt100 only.
        Put the terminal in cursor mode (instead of application mode).
        """
    self.write_raw('\x1b[?1l')