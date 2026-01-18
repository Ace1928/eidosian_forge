from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
@classmethod
def assert_valid_input(cls, tag: Any) -> None:
    """Check if valid input tag or document."""
    if not cls.is_tag(tag):
        raise TypeError(f"Expected a BeautifulSoup 'Tag', but instead received type {type(tag)}")