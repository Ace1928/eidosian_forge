import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import click
import colorama
import ray  # noqa: F401
def set_format(self, format_tmpl=None):
    if not format_tmpl:
        from ray.autoscaler._private.constants import LOGGER_FORMAT
        format_tmpl = LOGGER_FORMAT
    self._formatter = logging.Formatter(format_tmpl)