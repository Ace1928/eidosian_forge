from __future__ import annotations
from typing import TextIO
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.data_structures import Size
from prompt_toolkit.styles import Attrs
from .base import Output
from .color_depth import ColorDepth
from .flush_stdout import flush_stdout
def get_default_color_depth(self) -> ColorDepth:
    return ColorDepth.DEPTH_1_BIT