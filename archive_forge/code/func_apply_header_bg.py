import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
def apply_header_bg(self, value: Any) -> str:
    """
        If defined, apply the header background color to header text
        :param value: object whose text is to be colored
        :return: formatted text
        """
    if self.header_bg is None:
        return str(value)
    return ansi.style(value, bg=self.header_bg)