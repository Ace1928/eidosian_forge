from __future__ import annotations
from collections.abc import Mapping
from functools import partial
from typing import NamedTuple
from urllib.parse import parse_qsl, quote, unquote, urlparse
from ..log import get_logger
def maybe_sanitize_url(url, mask='**'):
    """Sanitize url, or do nothing if url undefined."""
    if isinstance(url, str) and '://' in url:
        return sanitize_url(url, mask)
    return url