from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
@staticmethod
def create_fake_parent(el: bs4.Tag) -> _FakeParent:
    """Create fake parent for a given element."""
    return _FakeParent(el)