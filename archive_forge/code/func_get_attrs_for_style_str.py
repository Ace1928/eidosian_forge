from __future__ import annotations
import itertools
import re
from enum import Enum
from typing import Hashable, TypeVar
from prompt_toolkit.cache import SimpleCache
from .base import (
from .named_colors import NAMED_COLORS
def get_attrs_for_style_str(self, style_str: str, default: Attrs=DEFAULT_ATTRS) -> Attrs:
    return self._merged_style.get_attrs_for_style_str(style_str, default)