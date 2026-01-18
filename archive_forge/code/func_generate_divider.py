import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
def generate_divider(self) -> str:
    """Generate divider row"""
    if self.divider_char is None:
        return ''
    return utils.align_left('', fill_char=self.divider_char, width=self.total_width())