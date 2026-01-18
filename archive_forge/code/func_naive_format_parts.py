from __future__ import annotations
import logging
import numbers
import os
import sys
from logging.handlers import WatchedFileHandler
from .utils.encoding import safe_repr, safe_str
from .utils.functional import maybe_evaluate
from .utils.objects import cached_property
def naive_format_parts(fmt):
    parts = fmt.split('%')
    for i, e in enumerate(parts[1:]):
        yield (None if not e or not parts[i - 1] else e[0])