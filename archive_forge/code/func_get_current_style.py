from __future__ import annotations
import xml.dom.minidom as minidom
from string import Formatter
from typing import Any
from .base import FormattedText, StyleAndTextTuples
def get_current_style() -> str:
    """Build style string for current node."""
    parts = []
    if name_stack:
        parts.append('class:' + ','.join(name_stack))
    if fg_stack:
        parts.append('fg:' + fg_stack[-1])
    if bg_stack:
        parts.append('bg:' + bg_stack[-1])
    return ' '.join(parts)