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
def render_annotation(text: TextType, style: StyleType, justify: 'JustifyMethod'='center') -> 'RenderResult':
    render_text = console.render_str(text, style=style, highlight=False) if isinstance(text, str) else text
    return console.render(render_text, options=render_options.update(justify=justify))