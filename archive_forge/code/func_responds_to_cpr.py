from __future__ import annotations
import sys
from typing import Any, TextIO
from prompt_toolkit.data_structures import Size
from .base import Output
from .color_depth import ColorDepth
from .vt100 import Vt100_Output
from .win32 import Win32Output
@property
def responds_to_cpr(self) -> bool:
    return False