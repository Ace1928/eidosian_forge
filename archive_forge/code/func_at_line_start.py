from __future__ import annotations
import re
import importlib.util
import sys
from typing import TYPE_CHECKING, Sequence
def at_line_start(self) -> bool:
    """
        Returns True if current position is at start of line.

        Allows for up to three blank spaces at start of line.
        """
    if self.offset == 0:
        return True
    if self.offset > 3:
        return False
    return self.rawdata[self.line_offset:self.line_offset + self.offset].strip() == ''