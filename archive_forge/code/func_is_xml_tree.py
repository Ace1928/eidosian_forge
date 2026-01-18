from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
@staticmethod
def is_xml_tree(el: bs4.Tag) -> bool:
    """Check if element (or document) is from a XML tree."""
    return bool(el._is_xml)