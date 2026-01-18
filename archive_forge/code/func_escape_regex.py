from __future__ import annotations
from difflib import SequenceMatcher
from typing import Iterable, Iterator
from kombu import version_info_t
def escape_regex(p, white=''):
    """Escape string for use within a regular expression."""
    return ''.join((c if c.isalnum() or c in white else '\\000' if c == '\x00' else '\\' + c for c in p))