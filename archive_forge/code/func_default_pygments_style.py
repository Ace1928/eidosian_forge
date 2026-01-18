from __future__ import annotations
from prompt_toolkit.cache import memoized
from .base import ANSI_COLOR_NAMES, BaseStyle
from .named_colors import NAMED_COLORS
from .style import Style, merge_styles
@memoized()
def default_pygments_style() -> Style:
    """
    Create a `Style` object that contains the default Pygments style.
    """
    return Style.from_dict(PYGMENTS_DEFAULT_STYLE)