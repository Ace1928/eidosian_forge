from __future__ import annotations
from typing import TYPE_CHECKING
from .style import Style
def style_from_pygments_dict(pygments_dict: dict[Token, str]) -> Style:
    """
    Create a :class:`.Style` instance from a Pygments style dictionary.
    (One that maps Token objects to style strings.)
    """
    pygments_style = []
    for token, style in pygments_dict.items():
        pygments_style.append((pygments_token_to_classname(token), style))
    return Style(pygments_style)