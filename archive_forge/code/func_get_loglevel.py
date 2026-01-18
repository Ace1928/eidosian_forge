from __future__ import annotations
import logging
import numbers
import os
import sys
from logging.handlers import WatchedFileHandler
from .utils.encoding import safe_repr, safe_str
from .utils.functional import maybe_evaluate
from .utils.objects import cached_property
def get_loglevel(self, level):
    if not isinstance(level, numbers.Integral):
        return LOG_LEVELS[level]
    return level