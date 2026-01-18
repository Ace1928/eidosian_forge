from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
@staticmethod
def is_cdata(obj: bs4.PageElement) -> bool:
    """Is CDATA."""
    return isinstance(obj, bs4.CData)