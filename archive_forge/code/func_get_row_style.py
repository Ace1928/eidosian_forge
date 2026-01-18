from dataclasses import dataclass, field, replace
from typing import (
from . import box, errors
from ._loop import loop_first_last, loop_last
from ._pick import pick_bool
from ._ratio import ratio_distribute, ratio_reduce
from .align import VerticalAlignMethod
from .jupyter import JupyterMixin
from .measure import Measurement
from .padding import Padding, PaddingDimensions
from .protocol import is_renderable
from .segment import Segment
from .style import Style, StyleType
from .text import Text, TextType
def get_row_style(self, console: 'Console', index: int) -> StyleType:
    """Get the current row style."""
    style = Style.null()
    if self.row_styles:
        style += console.get_style(self.row_styles[index % len(self.row_styles)])
    row_style = self.rows[index].style
    if row_style is not None:
        style += console.get_style(row_style)
    return style