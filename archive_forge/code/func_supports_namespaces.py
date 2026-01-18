from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def supports_namespaces(self) -> bool:
    """Check if namespaces are supported in the HTML type."""
    return self.is_xml or self.has_html_namespace