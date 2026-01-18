from __future__ import annotations
from prompt_toolkit.cache import memoized
from .base import ANSI_COLOR_NAMES, BaseStyle
from .named_colors import NAMED_COLORS
from .style import Style, merge_styles
@memoized()
def default_ui_style() -> BaseStyle:
    """
    Create a default `Style` object.
    """
    return merge_styles([Style(PROMPT_TOOLKIT_STYLE), Style(COLORS_STYLE), Style(WIDGETS_STYLE)])